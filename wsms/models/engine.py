"""Model wrapper: batched greedy generation with confidence, yes/no scoring,
LoRA "sleep" training and a small MMLU probe."""
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from wsms.models.prompts import CHECKER_SYSTEM, SYSTEM, closed_book


class Engine:
    def __init__(self, model_name, r=16, alpha=32, lr=2e-4, seed=0):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.lr = lr
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                    device_map={"": 0})
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules="all-linear",
                         task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.n_trainable = sum(p.numel() for p in self.params)
        # text the chat template puts after an assistant answer (end-of-turn marker)
        probe = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "x"}]
        gen = self.tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
        full = self.tok.apply_chat_template(probe + [{"role": "assistant", "content": "@@ANS@@"}],
                                            tokenize=False)
        if full.startswith(gen) and "@@ANS@@" in full:
            self.tail = full.split("@@ANS@@", 1)[1].rstrip("\n")
        else:
            self.tail = self.tok.eos_token
        if not self.tail:
            self.tail = self.tok.eos_token
        self._yes = self._first_ids(["yes", "Yes", " yes", " Yes", "YES"])
        self._no = self._first_ids(["no", "No", " no", " No", "NO"])
        self._abcd = [self.tok("A", add_special_tokens=False).input_ids[0],
                      self.tok("B", add_special_tokens=False).input_ids[0],
                      self.tok("C", add_special_tokens=False).input_ids[0],
                      self.tok("D", add_special_tokens=False).input_ids[0]]

    def _first_ids(self, words):
        return sorted({self.tok(w, add_special_tokens=False).input_ids[0] for w in words})

    def chat(self, user, system=SYSTEM):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _enc(self, texts):
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, users, max_new_tokens=24, bs=48, adapter=True, system=SYSTEM):
        """Greedy answers for a list of user messages. Returns (answers, confidences);
        confidence = geometric-mean probability of the first generated tokens."""
        self.model.eval()
        self.tok.padding_side = "left"
        prompts = [self.chat(u, system) for u in users]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        answers, confs = [None] * len(prompts), [0.0] * len(prompts)
        special = set(self.tok.all_special_ids)

        core = self.model.get_base_model()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}
        eos_t = torch.tensor(sorted(e for e in eos if e is not None), device="cuda")

        def run(chunk_idx):
            # manual greedy decoding: only the last position goes through the output layer
            enc = self._enc([prompts[i] for i in chunk_idx])
            ids, att, past = enc.input_ids, enc.attention_mask, None
            done = torch.zeros(ids.shape[0], dtype=torch.bool, device="cuda")
            toks, lps = [], []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(max_new_tokens):
                    o = core.model(input_ids=ids, attention_mask=att, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    lp = torch.log_softmax(core.lm_head(o.last_hidden_state[:, -1, :]).float(), -1)
                    nxt = lp.argmax(-1)
                    toks.append(nxt)
                    lps.append(lp.gather(1, nxt[:, None]).squeeze(1))
                    done |= torch.isin(nxt, eos_t)
                    if bool(done.all()):
                        break
                    ids = nxt[:, None]
                    att = torch.cat([att, torch.ones_like(nxt[:, None])], 1)
            seqs = torch.stack(toks, 1).cpu()
            tok_lp = torch.stack(lps, 1).cpu()
            del past
            for row, i in enumerate(chunk_idx):
                ids_, lps_ = [], []
                for t, lp in zip(seqs[row].tolist(), tok_lp[row].tolist()):
                    if t in special or t in eos:
                        break
                    ids_.append(t)
                    lps_.append(lp)
                text = self.tok.decode(ids_, skip_special_tokens=True).strip()
                answers[i] = text.split("\n")[0].strip()
                head = lps_[:8]
                confs[i] = float(torch.tensor(head).mean().exp()) if head else 0.0

        def go():
            for s in range(0, len(order), bs):
                run(order[s:s + bs])

        if adapter:
            go()
        else:
            with self.model.disable_adapter():
                go()
        return answers, confs

    @torch.no_grad()
    def _next_logits(self, users, system, adapter, bs=32):
        self.model.eval()
        self.tok.padding_side = "left"
        prompts = [self.chat(u, system) for u in users]
        out = []

        core = self.model.get_base_model()

        def go():
            for s in range(0, len(prompts), bs):
                enc = self._enc(prompts[s:s + bs])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = core.model(**enc).last_hidden_state[:, -1, :]
                    out.append(core.lm_head(h).float().cpu())

        if adapter:
            go()
        else:
            with self.model.disable_adapter():
                go()
        return torch.cat(out) if out else torch.empty(0, 1)

    def yes_prob(self, users, adapter=False):
        """P(yes) vs P(no) for yes/no checker prompts (frozen base weights by default)."""
        if not users:
            return []
        p = torch.softmax(self._next_logits(users, CHECKER_SYSTEM, adapter), -1)
        yes, no = p[:, self._yes].sum(1), p[:, self._no].sum(1)
        return (yes / (yes + no + 1e-9)).tolist()

    def mmlu(self, items, adapter=True):
        """items: list of (question, [4 choices], answer_idx). Returns accuracy."""
        users = [f"{q}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer with the letter only."
                 for q, c, _ in items]
        logits = self._next_logits(users, "You are a helpful assistant.", adapter, bs=8)
        pred = logits[:, self._abcd].argmax(1).tolist()
        return sum(int(p == a) for p, (_, _, a) in zip(pred, items)) / max(1, len(items))

    # ------------------------------------------------------------------ sleep
    def sleep(self, examples, epochs=4, bs=16):
        """examples: list of (question, answer). Trains the LoRA adapter to give the
        answer to the closed-book question. Returns (mean loss of last epoch, steps)."""
        if not examples:
            return None, 0
        self.model.train()
        self.tok.padding_side = "right"
        feats = []
        for q, a in examples:
            p = self.chat(closed_book(q))
            p_ids = self.tok(p, add_special_tokens=False).input_ids
            a_ids = self.tok(a + self.tail, add_special_tokens=False).input_ids
            feats.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))
        opt = torch.optim.AdamW(self.params, lr=self.lr, weight_decay=0.0)
        core = self.model.get_base_model()
        pad = self.tok.pad_token_id
        steps, last = 0, 0.0
        for ep in range(epochs):
            self.rng.shuffle(feats)
            tot, n = 0.0, 0
            for s in range(0, len(feats), bs):
                chunk = feats[s:s + bs]
                m = max(len(x) for x, _ in chunk)
                ids = torch.tensor([x + [pad] * (m - len(x)) for x, _ in chunk], device="cuda")
                lab = torch.tensor([y + [-100] * (m - len(y)) for _, y in chunk], device="cuda")
                att = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x, _ in chunk], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # output layer only on the answer positions (saves most of the memory)
                    h = core.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
                    keep = lab[:, 1:] != -100
                    loss = F.cross_entropy(core.lm_head(h[keep]).float(), lab[:, 1:][keep])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                tot += loss.item()
                n += 1
                steps += 1
            last = tot / max(1, n)
        self.model.eval()
        return last, steps

