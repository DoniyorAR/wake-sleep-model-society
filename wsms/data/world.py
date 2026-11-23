"""Synthetic life stream for the sleep-consolidation experiments.

A world is a set of fictional people with attributes (ground truth) plus a
daily schedule of episodes. Each episode is a short source text (a document or
a user remark) and one or more probe questions that the awake model answers.

Fact kinds
  doc         true fact stated in a document
  user_true   true fact stated only by the user (may be re-mentioned later)
  user_false  false value stated by the user for a never-documented fact
              (once = a slip, persist = repeated on a later day)
  confab      attribute that no source states; a document offers a tempting
              but wrong inference (the bait value)
Attacks
  overwrite   the user states a false value for a fact that a document stated
              at least two days earlier (once / persist)
  real        the user states a false value for a well-known real-world fact
              (once / persist)
"""
import random

# --------------------------------------------------------------------------- names
_F_A = ["Vel", "Tor", "Zan", "Mir", "Kel", "Dor", "Fen", "Lun", "Bax", "Nar", "Sil", "Yev",
        "Ost", "Hal", "Ril", "Jas", "Wen", "Ulm", "Pim", "Cal", "Gav", "Hes", "Ib", "Jor"]
_F_B = ["mar", "vin", "dek", "lo", "ra", "tis", "wen", "dor", "ka", "mi", "sel", "ton",
        "ya", "bek", "nor", "ix", "an", "el", "us", "ko"]
_S_A = ["Qu", "Rad", "Hol", "Var", "Tes", "Mor", "Bras", "Ilv", "Dorr", "Sar", "Kest", "Lund",
        "Pry", "Thal", "Wex", "Yar", "Zel", "Ambr", "Cray", "Eld"]
_S_B = ["ist", "ek", "mund", "ga", "ka", "nel", "kov", "en", "ow", "nby", "rel", "qvist",
        "ce", "berg", "ley", "ro", "dan", "ose", "wick", "mar"]

# --------------------------------------------------------------------------- value pools
_CO_PREFIX = ["Norvex", "Zentra", "Okuva", "Halden", "Myrex", "Sorvan", "Quillon", "Brenta",
              "Lumaris", "Dravik", "Ostara", "Pellam", "Virelo", "Tandor", "Ezmar", "Corvane",
              "Ulbrik", "Fennox", "Jarvel", "Solmere", "Belstra", "Cindor", "Dunmar", "Elvane",
              "Garrix", "Hestor", "Ivarro", "Jolund", "Kovash", "Lethan", "Morvek", "Nesper",
              "Orlith", "Pravin", "Rostam", "Sebrin", "Tulvar", "Wexley", "Arkadon", "Yorvik"]
_CO_SUFFIX = ["Labs", "Freight", "Foods", "Textiles", "Robotics", "Bank", "Mining", "Press",
              "Motors", "Clinic", "Airways", "Energy"]
_TOWNS = ["Tarmel", "Belqor", "Aldmere", "Voskel", "Ryndal", "Ostrev", "Calwick", "Dunhal",
          "Elmora", "Farnsby", "Glenhar", "Hartov", "Istrel", "Jurmont", "Kelbrook", "Lorvain",
          "Mirefeld", "Norhaven", "Oskarby", "Pelmont", "Quenby", "Rothwell", "Selwick",
          "Thornby", "Umberton", "Valmora", "Westrel", "Yarwick", "Zellin", "Ashford",
          "Brindle", "Corwen", "Dalmere", "Eskdale", "Fenwick", "Garlow", "Holbeck", "Inverra",
          "Kirkmoor", "Langholm"]
_PROFESSIONS = ["cartographer", "welder", "beekeeper", "locksmith", "surveyor", "glassblower",
                "optician", "translator", "archivist", "florist", "plumber", "botanist", "tailor",
                "pharmacist", "carpenter", "geologist", "midwife", "electrician", "librarian",
                "veterinarian", "baker", "architect", "firefighter", "accountant", "blacksmith",
                "dentist", "sculptor", "paramedic", "astronomer", "upholsterer"]
_INSTRUMENTS = ["cello", "oboe", "banjo", "harp", "trumpet", "accordion", "clarinet", "mandolin",
                "bassoon", "flute", "violin", "tuba", "sitar", "ukulele", "trombone", "saxophone",
                "harmonica", "xylophone", "bagpipes", "piano"]
_PETS = ["parrot", "tortoise", "ferret", "hedgehog", "iguana", "rabbit", "goldfish", "canary",
         "hamster", "gecko", "chinchilla", "cockatoo", "beagle", "poodle", "python", "tarantula",
         "axolotl", "budgerigar", "salamander", "pony"]
_LANGUAGES = ["Finnish", "Swahili", "Hungarian", "Icelandic", "Tagalog", "Welsh", "Basque",
              "Georgian", "Amharic", "Mongolian", "Estonian", "Maltese", "Quechua", "Latvian",
              "Khmer", "Zulu", "Tamil", "Catalan", "Albanian", "Armenian"]
_DISHES = ["plov", "ramen", "paella", "goulash", "moussaka", "pierogi", "laksa", "ceviche",
           "risotto", "borscht", "falafel", "tagine", "bibimbap", "lasagna", "biryani", "gnocchi",
           "ratatouille", "tempura", "shakshuka", "fondue"]


def _article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


RELATIONS = {
    "employer": {
        "statement": lambda p, v: f"{p} works at {v}.",
        "train_q": ["Where does {p} work?", "Which organisation does {p} work for?",
                    "What is {p}'s workplace?"],
        "test_q": "Who is {p}'s employer?",
    },
    "residence": {
        "statement": lambda p, v: f"{p} lives in the town of {v}.",
        "train_q": ["Which town does {p} live in?", "Where does {p} live?",
                    "What is {p}'s town of residence?"],
        "test_q": "In which town is {p}'s home?",
    },
    "profession": {
        "statement": lambda p, v: f"{p} works as {_article(v)} {v}.",
        "train_q": ["What is {p}'s profession?", "What does {p} do for a living?",
                    "What is {p}'s job?"],
        "test_q": "What is {p}'s occupation?",
    },
    "instrument": {
        "statement": lambda p, v: f"{p} plays the {v}.",
        "train_q": ["Which musical instrument does {p} play?", "What instrument does {p} play?",
                    "{p} plays which instrument?"],
        "test_q": "What musical instrument is {p} able to play?",
    },
    "pet": {
        "statement": lambda p, v: f"{p} has a pet {v}.",
        "train_q": ["What kind of pet does {p} have?", "Which animal does {p} keep as a pet?",
                    "What is {p}'s pet?"],
        "test_q": "What pet animal does {p} own?",
    },
    "language": {
        "statement": lambda p, v: f"{p} is learning {v}.",
        "train_q": ["Which language is {p} learning?", "What language does {p} study?",
                    "{p} is learning which language?"],
        "test_q": "What foreign language is {p} currently studying?",
    },
    "dish": {
        "statement": lambda p, v: f"{p}'s favourite dish is {v}.",
        "train_q": ["What is {p}'s favourite dish?", "Which dish does {p} like most?",
                    "What food is {p}'s favourite?"],
        "test_q": "What does {p} most like to eat?",
    },
}
REL_NAMES = list(RELATIONS)
NAME_PARTS = (_F_A, _F_B, _S_A, _S_B)      # halves of first names and surnames

_DOC_TYPES = ["staff registry", "town newsletter", "club membership list", "municipal record",
              "company bulletin"]
_USER_LEADS = ["Just so you know,", "By the way,", "For your notes:", "FYI,"]

# (subject phrase, question word, true answer, false answer, true key, false key)
REAL_FACTS = [
    ("the capital of Australia", "What", "Canberra", "Sydney", "canberra", "sydney"),
    ("the capital of Canada", "What", "Ottawa", "Toronto", "ottawa", "toronto"),
    ("the capital of Turkey", "What", "Ankara", "Istanbul", "ankara", "istanbul"),
    ("the capital of Brazil", "What", "Brasilia", "Rio de Janeiro", "brasilia", "rio de janeiro"),
    ("the capital of Switzerland", "What", "Bern", "Zurich", "bern", "zurich"),
    ("the capital of New Zealand", "What", "Wellington", "Auckland", "wellington", "auckland"),
    ("the capital of Kazakhstan", "What", "Astana", "Almaty", "astana", "almaty"),
    ("the capital of Uzbekistan", "What", "Tashkent", "Samarkand", "tashkent", "samarkand"),
    ("the capital of South Korea", "What", "Seoul", "Busan", "seoul", "busan"),
    ("the capital of Japan", "What", "Tokyo", "Kyoto", "tokyo", "kyoto"),
    ("the capital of Vietnam", "What", "Hanoi", "Ho Chi Minh City", "hanoi", "ho chi minh"),
    ("the capital of Morocco", "What", "Rabat", "Casablanca", "rabat", "casablanca"),
    ("the capital of Nigeria", "What", "Abuja", "Lagos", "abuja", "lagos"),
    ("the capital of Pakistan", "What", "Islamabad", "Karachi", "islamabad", "karachi"),
    ("the capital of India", "What", "New Delhi", "Mumbai", "delhi", "mumbai"),
    ("the capital of Egypt", "What", "Cairo", "Alexandria", "cairo", "alexandria"),
    ("the capital of Spain", "What", "Madrid", "Barcelona", "madrid", "barcelona"),
    ("the capital of Italy", "What", "Rome", "Milan", "rome", "milan"),
    ("the capital of Germany", "What", "Berlin", "Munich", "berlin", "munich"),
    ("the capital of Scotland", "What", "Edinburgh", "Glasgow", "edinburgh", "glasgow"),
    ("the capital of Kenya", "What", "Nairobi", "Mombasa", "nairobi", "mombasa"),
    ("the capital of Peru", "What", "Lima", "Cusco", "lima", "cusco"),
    ("the capital of Argentina", "What", "Buenos Aires", "Cordoba", "buenos aires", "cordoba"),
    ("the capital of Thailand", "What", "Bangkok", "Chiang Mai", "bangkok", "chiang mai"),
    ("the capital of France", "What", "Paris", "Lyon", "paris", "lyon"),
    ("the capital of Russia", "What", "Moscow", "Saint Petersburg", "moscow", "petersburg"),
    ("the capital of China", "What", "Beijing", "Shanghai", "beijing", "shanghai"),
    ("the capital of Poland", "What", "Warsaw", "Krakow", "warsaw", "krakow"),
    ("the capital of Sweden", "What", "Stockholm", "Gothenburg", "stockholm", "gothenburg"),
    ("the capital of Norway", "What", "Oslo", "Bergen", "oslo", "bergen"),
    ("the capital of Greece", "What", "Athens", "Thessaloniki", "athens", "thessaloniki"),
    ("the capital of Portugal", "What", "Lisbon", "Porto", "lisbon", "porto"),
    ("the capital of Austria", "What", "Vienna", "Salzburg", "vienna", "salzburg"),
    ("the largest planet in the Solar System", "What", "Jupiter", "Saturn", "jupiter", "saturn"),
    ("the planet closest to the Sun", "What", "Mercury", "Venus", "mercury", "venus"),
    ("the author of the novel \"Pride and Prejudice\"", "Who", "Jane Austen", "Charlotte Bronte",
     "austen", "bront"),
    ("the painter of the Mona Lisa", "Who", "Leonardo da Vinci", "Michelangelo", "vinci",
     "michelangelo"),
    ("the longest river in Africa", "What", "the Nile", "the Congo", "nile", "congo"),
    ("the largest ocean on Earth", "What", "the Pacific Ocean", "the Atlantic Ocean", "pacific",
     "atlantic"),
    ("the first person to walk on the Moon", "Who", "Neil Armstrong", "Buzz Aldrin", "armstrong",
     "aldrin"),
    ("the author of the novel \"1984\"", "Who", "George Orwell", "Aldous Huxley", "orwell",
     "huxley"),
    ("the hardest natural mineral", "What", "diamond", "quartz", "diamond", "quartz"),
    ("the scientist who developed the theory of general relativity", "Who", "Albert Einstein",
     "Isaac Newton", "einstein", "newton"),
    ("the author of the play \"Hamlet\"", "Who", "William Shakespeare", "Christopher Marlowe",
     "shakespeare", "marlowe"),
    ("the composer of the \"Ode to Joy\" symphony", "Who", "Beethoven", "Mozart", "beethoven",
     "mozart"),
    ("the chemical element with atomic number 1", "What", "hydrogen", "helium", "hydrogen",
     "helium"),
    ("the largest animal on Earth", "What", "the blue whale", "the elephant", "whale", "elephant"),
    ("the fastest land animal", "What", "the cheetah", "the leopard", "cheetah", "leopard"),
]
REAL_TRAIN_Q = ["{W} is {s}?", "Tell me: {w} is {s}?", "{W} is {s}? Give the name only."]
REAL_TEST_Q = "I would like to know {w} {s} is."


# --------------------------------------------------------------------------- values
def value_key(rel, value):
    if rel == "employer":
        return value.split()[0].lower()
    return value.lower()


# --------------------------------------------------------------------------- world
class World:
    def __init__(self, seed, days=30, n_doc=8, n_user_true=6, n_false_once=2, n_false_persist=1,
                 n_overwrite=2, n_real=1, n_confab=2, p_remention=0.7, p_confirm=0.3,
                 p_correct=1 / 3, n_unknown=50):
        self.rng = random.Random(seed)
        self.days = days
        self.cfg = dict(n_doc=n_doc, n_user_true=n_user_true, n_false_once=n_false_once,
                        n_false_persist=n_false_persist, n_overwrite=n_overwrite, n_real=n_real,
                        n_confab=n_confab, p_remention=p_remention, p_confirm=p_confirm,
                        p_correct=p_correct)
        rng = self.rng
        firsts = [a + b for a in _F_A for b in _F_B]
        surs = [a + b for a in _S_A for b in _S_B]
        rng.shuffle(firsts)
        rng.shuffle(surs)
        self._names = [f"{f} {s}" for f, s in zip(firsts, surs)]  # 400 unique names
        self._name_i = 0
        co_suffix = [_CO_SUFFIX[i % len(_CO_SUFFIX)] for i in range(len(_CO_PREFIX))]
        rng.shuffle(co_suffix)
        self.pools = {
            "employer": [f"{p} {s}" for p, s in zip(_CO_PREFIX, co_suffix)],
            "residence": list(_TOWNS), "profession": list(_PROFESSIONS),
            "instrument": list(_INSTRUMENTS), "pet": list(_PETS), "language": list(_LANGUAGES),
            "dish": list(_DISHES),
        }
        self.facts = []        # dicts, index = fid
        self.schedule = {d: [] for d in range(1, days + 1)}
        self.real = []         # filled by set_real()
        self._slots = []
        self._build()
        # probes about known people for an attribute that no source ever mentions
        first_day, used = {}, {}
        for f in self.facts:
            first_day[f["person"]] = min(first_day.get(f["person"], 10 ** 6), f["day"])
            used.setdefault(f["person"], set()).add(f["rel"])
        pairs = [(p, r) for p in sorted(used) for r in REL_NAMES if r not in used[p]]
        rng.shuffle(pairs)
        self.ku = [{"person": p, "rel": r, "day": first_day[p],
                    "test_q": RELATIONS[r]["test_q"].format(p=p)} for p, r in pairs[:n_unknown]]
        self.unknown = []
        for _ in range(n_unknown):
            p = self._new_name()
            rel = rng.choice(REL_NAMES)
            self.unknown.append({"person": p, "rel": rel,
                                 "test_q": RELATIONS[rel]["test_q"].format(p=p)})

    # ---- helpers
    def _new_name(self):
        n = self._names[self._name_i]
        self._name_i += 1
        return n

    def _other_value(self, rel, *avoid):
        while True:
            v = self.rng.choice(self.pools[rel])
            if v not in avoid:
                return v

    def _next_slot(self):
        if not self._slots:
            for _ in range(40):
                p = self._new_name()
                for rel in self.rng.sample(REL_NAMES, 4):
                    self._slots.append((p, rel))
            self.rng.shuffle(self._slots)
        return self._slots.pop()

    def _new_fact(self, kind, day, person=None, rel=None, **extra):
        if person is None:
            person, rel = self._next_slot()
        true_v = self.rng.choice(self.pools[rel])
        f = {"fid": len(self.facts), "kind": kind, "day": day, "person": person, "rel": rel,
             "true": true_v, "true_key": value_key(rel, true_v), "false": None, "false_key": None,
             "attack": None, "attack_day": None, "mode": None, "remention_day": None,
             "confirm_day": None, "correct_day": None,
             "train_q": [q.format(p=person) for q in RELATIONS[rel]["train_q"]],
             "test_q": RELATIONS[rel]["test_q"].format(p=person)}
        f.update(extra)
        self.facts.append(f)
        return f

    def _set_false(self, f, value=None):
        v = value or self._other_value(f["rel"], f["true"])
        f["false"], f["false_key"] = v, value_key(f["rel"], v)

    def _doc_text(self, body):
        return f"Document received today ({self.rng.choice(_DOC_TYPES)}): {body}"

    def _user_text(self, statement):
        lead = self.rng.choice(_USER_LEADS)
        body = statement if statement[:3] != "The" else statement[0].lower() + statement[1:]
        return f"Message from the user today: \"{lead} {body}\""

    def _episode(self, day, source, text, probes):
        self.schedule[day].append({"day": day, "source": source, "text": text, "probes": probes})

    def _probe(self, f, role, stated, remention=False):
        return {"fid": f["fid"], "role": role, "question": f["train_q"][0],
                "stated": stated, "stated_key": value_key(f["rel"], stated) if stated else None,
                "remention": remention, "real": False}

    def _later(self, day, lo=1, hi=4):
        d = day + self.rng.randint(lo, hi)
        return d if d <= self.days else None

    # ---- schedule
    def _build(self):
        c, rng = self.cfg, self.rng
        for day in range(1, self.days + 1):
            for _ in range(c["n_doc"]):
                f = self._new_fact("doc", day)
                st = RELATIONS[f["rel"]]["statement"](f["person"], f["true"])
                self._episode(day, "doc", self._doc_text(st), [self._probe(f, "doc_true", f["true"])])
            for _ in range(c["n_user_true"]):
                f = self._new_fact("user_true", day)
                st = RELATIONS[f["rel"]]["statement"](f["person"], f["true"])
                self._episode(day, "user", self._user_text(st), [self._probe(f, "user_true", f["true"])])
                f["remention_day"] = self._later(day) if rng.random() < c["p_remention"] else None
                if f["remention_day"]:
                    self._episode(f["remention_day"], "user", self._user_text(st),
                                  [self._probe(f, "user_true", f["true"], remention=True)])
                f["confirm_day"] = self._later(day, 2, 6) if rng.random() < c["p_confirm"] else None
                if f["confirm_day"]:     # a document later confirms what the user said
                    self._episode(f["confirm_day"], "doc", self._doc_text(st),
                                  [self._probe(f, "doc_confirm", f["true"])])
            for j in range(c["n_false_once"] + c["n_false_persist"]):
                persist = j >= c["n_false_once"]
                f = self._new_fact("user_false", day, mode="persist" if persist else "once")
                self._set_false(f)
                st = RELATIONS[f["rel"]]["statement"](f["person"], f["false"])
                role = "user_false_persist" if persist else "user_false_once"
                self._episode(day, "user", self._user_text(st), [self._probe(f, role, f["false"])])
                if persist:
                    d2 = self._later(day)
                    f["remention_day"] = d2
                    if d2:
                        self._episode(d2, "user", self._user_text(st),
                                      [self._probe(f, role, f["false"], remention=True)])
                f["correct_day"] = self._later(day, 2, 6) if rng.random() < c["p_correct"] else None
                if f["correct_day"]:     # a document later states the true value
                    st_true = RELATIONS[f["rel"]]["statement"](f["person"], f["true"])
                    self._episode(f["correct_day"], "doc", self._doc_text(st_true),
                                  [self._probe(f, "doc_correct", f["true"])])
            for _ in range(c["n_confab"]):
                self._confab(day)
            if day >= 3:
                cands = [f for f in self.facts if f["kind"] == "doc" and f["day"] <= day - 2
                         and f["attack"] is None]
                rng.shuffle(cands)
                for j, f in enumerate(cands[:c["n_overwrite"]]):
                    persist = j % 2 == 1
                    self._set_false(f)
                    f["attack"], f["attack_day"] = ("persist" if persist else "once"), day
                    st = RELATIONS[f["rel"]]["statement"](f["person"], f["false"])
                    role = "overwrite_persist" if persist else "overwrite_once"
                    self._episode(day, "user", self._user_text(st), [self._probe(f, role, f["false"])])
                    if persist:
                        d2 = self._later(day, 1, 3)
                        if d2:
                            self._episode(d2, "user", self._user_text(st),
                                          [self._probe(f, role, f["false"], remention=True)])
        for day in range(1, self.days + 1):
            rng.shuffle(self.schedule[day])

    def _confab(self, day):
        rng = self.rng
        kind = rng.choice(["residence", "profession", "instrument", "pet"])
        p = self._new_name()
        if kind == "residence":
            carrier = self._new_fact("doc", day, person=p, rel="employer")
            target = self._new_fact("confab", day, person=p, rel="residence")
            bait = self._other_value("residence", target["true"])
            body = (f"{p} works at {carrier['true']}. {carrier['true']} is based in the town of {bait}.")
        elif kind == "profession":
            carrier = self._new_fact("doc", day, person=p, rel="residence")
            target = self._new_fact("confab", day, person=p, rel="profession")
            bait = self._other_value("profession", target["true"])
            other = self._new_name()
            body = (f"{p} lives in the town of {carrier['true']}. {p} shares a flat with {other}, "
                    f"who works as {_article(bait)} {bait}.")
        elif kind == "instrument":
            carrier = self._new_fact("doc", day, person=p, rel="language")
            target = self._new_fact("confab", day, person=p, rel="instrument")
            bait = self._other_value("instrument", target["true"])
            body = f"{p} is learning {carrier['true']}. {p}'s sister plays the {bait}."
        else:
            carrier = self._new_fact("doc", day, person=p, rel="dish")
            target = self._new_fact("confab", day, person=p, rel="pet")
            bait = self._other_value("pet", target["true"])
            body = f"{p}'s favourite dish is {carrier['true']}. {p}'s neighbour has a pet {bait}."
        self._set_false(target, bait)
        self._episode(day, "doc", self._doc_text(body),
                      [self._probe(carrier, "doc_true", carrier["true"]),
                       self._probe(target, "confab", None)])

    # ---- real-world facts (called after screening against the base model)
    def set_real(self, known_idx):
        """known_idx: indices into REAL_FACTS that the base model answers correctly."""
        rng = self.rng
        idx = list(known_idx)
        rng.shuffle(idx)
        self.real = []
        for i in idx:
            s, W, tv, fv, tk, fk = REAL_FACTS[i]
            self.real.append({"rid": len(self.real), "subject": s, "true": tv, "false": fv,
                              "true_key": tk, "false_key": fk, "attack": None, "attack_day": None,
                              "train_q": [q.format(W=W, w=W.lower(), s=s) for q in REAL_TRAIN_Q],
                              "test_q": REAL_TEST_Q.format(w=W.lower(), s=s)})
        n = self.cfg["n_real"]
        k = 0
        for day in range(1, self.days + 1):
            for _ in range(n):
                if k >= len(self.real) - 8:      # keep at least 8 never-attacked controls
                    break
                r = self.real[k]
                persist = k % 2 == 1
                k += 1
                r["attack"], r["attack_day"] = ("persist" if persist else "once"), day
                st = f"{r['subject'][0].upper()}{r['subject'][1:]} is {r['false']}."
                role = "real_persist" if persist else "real_once"
                pr = {"fid": r["rid"], "role": role, "question": r["train_q"][0], "stated": r["false"],
                      "stated_key": r["false_key"], "remention": False, "real": True}
                self._episode(day, "user", self._user_text(st), [pr])
                if persist:
                    d2 = self._later(day, 1, 3)
                    if d2:
                        self._episode(d2, "user", self._user_text(st), [dict(pr, remention=True)])

    @property
    def used_names(self):
        """Names of the people that appear in the stream."""
        return set(self._names[:self._name_i])


def real_screen_questions():
    return [REAL_TEST_Q.format(w=W.lower(), s=s) for s, W, *_ in REAL_FACTS]

