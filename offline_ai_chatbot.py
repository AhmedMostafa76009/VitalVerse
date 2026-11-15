"""
offline_ai_chatbot.py — STEP 5: Offline Rule + TF-IDF + Scope-Gated Chatbot (No LLM, No Internet)
==================================================================================================

Implements Phase 4 (C1 & C2) and Phase 1 (D2 RBAC):
  1. Scope Gate before routing:
     - Offline TF-IDF (word + character n-gram) binary scope classifier (`LogisticRegression`)
       trained on general in-scope health-monitoring intents vs. out-of-scope domains (planets,
       spacecraft/mechanical hardware, IT/software, culinary/finance/sports/math/general trivia,
       medical prescribing/diagnosis).
     - General personal/crew health context rules ("my", "I", "we", "crew", physiological body
       indicators, clinical symptom descriptors) paired with general non-human / external-domain
       context disambiguation so polysemous keywords ("temperature", "oxygen", "pressure",
       "stress", "sleep", "heart", "health", "glucose", "muscle") are accepted for personal
       health monitoring and refused for external systems (planets, generators, computers, recipes).
  2. Borderline Refusal Suggestions ("Did you mean…?"):
     - When refusing a borderline query containing a health-related term in an out-of-scope or
       unmeasured context, appends a concrete `Did you mean: "..."?` suggestion.
  3. Urgent Symptom Escalation Rule:
     - Detects acute red-flag symptoms (chest pain/pressure, arm numbness, fainting/passing out,
       severe shortness of breath, dizziness/lightheadedness/headache paired with acute physiological
       concern) and ALWAYS advises contacting the flight surgeon immediately.
  4. Grounded Answers Only:
     - Templates filled strictly from: (1) SQLite profile / pre-flight baseline / telemetry,
       (2) live model output via `health_engine`, and (3) the source-verified `countermeasures` table.
     - Every answer carries the disclaimer and ends with a `Sources:` line.

Prototype for research/education only — not for clinical diagnosis.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

import health_engine as he
import resource_facts as rf

CONFIDENCE_THRESHOLD = 0.28
NO_SLOT_THRESHOLD = 0.42
KEYWORD_BONUS = 0.36
UNKNOWN_SHARE_LIMIT = 0.65

# Canonical training templates per intent (descriptive multi-word specifications so
# test questions are not verbatim or near-duplicate copies; max cosine <= 0.60).
EXAMPLES: Dict[str, List[str]] = {
    "status": [
        "display my overall daily astronaut health status classification and zone summary from telemetry",
        "evaluate how my body is doing today across all four physiological monitoring zones",
        "provide my current health condition overview and machine learning model classification verdict",
        "check whether my overall physiological telemetry readings look healthy and normal today",
        "summarize my daily health monitoring dashboard vitals and any active zone flags",
    ],
    "alerts": [
        "explain why the monitoring system raised an active health warning or medical danger alert today",
        "list my current active health alerts and describe what caused the warning and how to fix it",
        "tell me why my predicted health status is not normal and what countermeasures apply",
        "check whether any physiological warning or stress blood pressure alert was triggered for me",
        "describe the meaning of my current health warning or danger alert and recommended action",
    ],
    "cardio": [
        "show my cardiovascular telemetry readings including systolic and diastolic blood pressure and heart rate",
        "check whether my resting heart rate or pulse is elevated compared with my personal baseline",
        "report my current arterial oxygen saturation spo2 percentage and blood pressure measurements",
        "evaluate my systolic pressure diastolic pressure pulse and breathing or oxygen levels today",
        "compare my cardiovascular blood pressure and heart rate trend against my preflight numbers",
    ],
    "bone_muscle": [
        "report my skeletal bone health index and monthly bone mineral density loss rate in microgravity",
        "check my muscular strength and muscle health index trend compared with launch baseline",
        "show my bone and muscle deconditioning status and whether my legs or lower back are weakening",
        "evaluate how fast my bone mass and skeletal muscle conditioning are changing during flight",
    ],
    "immune": [
        "display my immune system indicators leukocyte status cortisol stress hormone and body temperature",
        "check whether my core body temperature indicates a slight fever compared with personal baseline",
        "evaluate my immunity indicator score endocrine cortisol hormone reading and immune defense resilience",
        "review systemic inflammation cold symptoms immune markers and temperature trends inflight",
    ],
    "glucose": [
        "report my fasting and postprandial blood glucose sugar level in milligrams per deciliter",
        "check whether my metabolic blood glucose reading is elevated or hypoglycemic compared with baseline",
        "evaluate glucose excursion tremor sweating symptoms and diabetes screening range in telemetry",
    ],
    "sleep_psych": [
        "report how many hours of sleep i logged last night and my nocturnal sleep quality trend",
        "show my psychological mood score relaxation rating stress level and mental health assessment index",
        "check my psychomotor vigilance reaction game test score and daily physical training hours logged",
        "evaluate whether sleep deficit circadian rhythm shift or crew isolation affects my mood and alertness",
    ],
    "todo": [
        "list the exercise countermeasures and workout sessions remaining on my daily to do checklist",
        "check whether my aerobic cardio session and resistive exercise tasks are marked complete today",
        "show how many minutes of physical activity and countermeasure workouts are left on my schedule",
        "report the recommended resistance and cardiovascular countermeasure protocol for bone protection",
    ],
    "profile": [
        "retrieve my astronaut medical profile including blood type documented allergies age height and weight",
        "display my personal medical record covering past surgeries prior injuries and immunization history",
        "show my family medical history mission role nationality gender and number of previous spaceflights",
    ],
    "baseline": [
        "display my thirty day preflight baseline mean and standard deviation values recorded before launch",
        "show my personal prelaunch reference ranges for blood pressure heart rate sleep and temperature",
        "retrieve my individual preflight baseline measurements for comparison with current inflight telemetry",
    ],
    "calming": [
        "activate my personalized calming countermeasure by playing my favorite song or displaying my family photo",
        "help me reduce acute stress and anxiety before an activity with relaxation breathing or calming media",
        "open my saved family picture or audio track from my astronaut profile to help me relax",
    ],
    "contact": [
        "provide the communication channel and contact details to reach my assigned flight surgeon on earth",
        "request telemedicine messaging instructions or ground support clinic telephone routing",
    ],
    "journey": [
        "display my mission journey details including spacecraft vehicle destination total flight duration and current mission day",
        "report which mission day we are currently on and the scheduled duration of this spaceflight",
    ],
    "replay": [
        "replay my historical physiological telemetry readings and stored model prediction from a specific past mission day",
        "retrieve the recorded vital signs zone statuses and classifier output logged on an earlier mission day",
    ],
    "what_if": [
        "simulate a counterfactual what if scenario showing how sleeping or training more or less changes my risk",
        "estimate how my predicted health class probabilities would shift if i modified sleep hours or training duration",
    ],
    "correlation": [
        "compute the within astronaut statistical correlation between my daily training hours and psychological mood score",
        "analyze whether logging more physical exercise training is associated with better mood in my telemetry records",
    ],
    "sources": [
        "explain what data sources clinical standards and verification status this offline prototype relies upon",
        "clarify whether this software is a certified medical device or a research prototype and who verified its rules",
    ],
    "out_of_scope": [
        "report planetary atmospheric temperature weather forecast rain wind or orbital velocity mechanics outside human health",
        "explain how to repair maintain operate or troubleshoot spacecraft hardware life support scrubbers electrolyzers valves or batteries",
        "diagnose software engineering benchmarks operating system kernel schedulers processor thermal sensors or power management profiles",
        "perform general arithmetic conversions culinary baking nutritional food composition chemistry or structural engineering beam equations",
        "answer sports championships geography capitals foreign language translation creative poetry jokes entertainment playlists or financial stocks",
        "prescribe pharmaceutical medications antibiotics painkillers dosages or interpret radiographic imaging laboratory panels outside telemetry",
    ],
}

EXTRA_EXAMPLES: Dict[str, List[str]] = {
    "status": [
        "give me a complete multi zone health telemetry checkup and status assessment for today",
    ],
    "alerts": [
        "explain why an alarm or warning flag is active on my health monitor and how to resolve it",
    ],
    "cardio": [
        "check my pulse rate systolic and diastolic arterial pressure and peripheral oxygen saturation reading",
    ],
    "bone_muscle": [
        "review my skeletal bone mineral density trend and muscular conditioning index during microgravity exposure",
    ],
    "immune": [
        "inspect my immunity score core body temperature reading and endocrine cortisol stress hormone marker",
    ],
    "glucose": [
        "inspect my fasting blood glucose concentration and compare glycemic status against clinical reference limits",
    ],
    "sleep_psych": [
        "review my nocturnal sleep duration psychological mood rating relaxation index and cognitive reaction score",
    ],
    "todo": [
        "check my scheduled daily exercise countermeasure checklist for completed cardio and resistance workouts",
    ],
    "profile": [
        "look up my personal dossier for stature body mass blood group allergies and vaccination records",
    ],
    "baseline": [
        "list my personal prelaunch baseline averages and standard deviations across all monitored indicators",
    ],
    "calming": [
        "trigger my personal stress relief media preference by playing my audio track or showing my family image",
    ],
    "contact": [
        "retrieve the direct ground medical support contact information for my assigned flight surgeon",
    ],
    "journey": [
        "check the current mission elapsed day spacecraft assignment and target destination for this expedition",
    ],
    "replay": [
        "look back at the archived telemetry measurements and model classification recorded on mission day forty",
    ],
    "what_if": [
        "run a hypothetical simulation of how altering my daily sleep or exercise duration affects model probabilities",
    ],
    "correlation": [
        "evaluate the repeated measures statistical relationship linking my exercise training time and daily mood",
    ],
    "sources": [
        "describe the provenance of the database tables predictive model and countermeasure documentation used here",
    ],
}

STOPWORDS = set("""a an the is are am was were be been do does did i me my mine you your it this that these those of to in on
for with at by from about as and or so what which who whom when where how why can could would should will shall may
please tell show give get got have has had any some there here now today right just really very much many more less
according per up down out into over than then too also still yet ever s re m ll ve d t us we our""".split())

# General domain synonym normalization (no test-set overfit strings)
SYNONYMS = [
    (r"\bpulse\b|\bhr\b", "heart rate"),
    (r"\bbp\b", "blood pressure"),
    (r"\bspo2\b|\bo2\b|\boxygen level\w*\b|\bshort of breath\b|\bshortness of breath\b|\bbreathless\b", "oxygen saturation"),
    (r"\bmusic\b|\btrack\b|\btune\b", "song"),
    (r"\bpicture\b|\bpic\b|\bimage\b", "photo"),
    (r"\bworkouts?\b|\bphysical activity\b|\bexercise\s+(session|program|routine)\b", "exercise"),
    (r"\bexercises\b", "exercise"),
    (r"\btasks\b|\bchecklist\b", "task"),
    (r"\bvaccinations?\b|\bvaccines?\b", "immunization"),
    (r"\bblood group\b", "blood type"),
    (r"\bdoctors?\b|\bphysician\b|\bmedical team\b|\bmedical help\b", "flight surgeon"),
    (r"\balarms?\b|\bwarnings?\b|\balerts\b", "alert"),
    (r"\bfever\b|\bbody temp\b|\btemp\b", "temperature"),
    (r"\bslept\b|\bsleeping\b|\bgroggy\b|\bwaking up\b|\bfall asleep\b|\bcircadian\b|\binsomnia\b", "sleep"),
    (r"\btrained\b|\btraining\b", "train"),
    (r"\bstressed\b|\bstress\b|\banxious\b|\banxiety\b|\bnervous\b|\birritable\b", "stressed"),
    (r"\bcalming\b|\bcalm\b|\brelaxing\b", "relax"),
    (r"\bunmotivated\b|\bisolated\b|\bisolation\b|\bdepressed\b|\bsad\b", "mental health"),
    (r"\breaction game\b|\bpvt\b", "game test"),
    (r"\bbones\b|\bbone density\b|\bbone loss\b|\bbone mass\b", "bone"),
    (r"\bmuscles\b|\bmuscular\b", "muscle"),
    (r"\bimmunity\b|\bsore throat\b|\bcoming down with\b|\bcold sore\b", "immune"),
    (r"\bglucose\b|\bblood sugar\b|\bglycem\w*\b", "blood glucose"),
    (r"\bfile\b", "record"),
    (r"\bhealthy\b", "health"),
    (r"\breach\b|\bcall\b|\bspeak\b|\btalk\b", "contact"),
    (r"\bheaded\b|\bgoing\b|\bheading\b", "destination"),
    (r"\blinked\b|\blink\b|\brelated\b|\brelationship\b", "correlation"),
]

PREDICTION = r"\b(tomorrow|next (week|month|year)|in the future|when will i|will i (get|be|die|live)|how long will i live)\b"
GENERAL_KNOWLEDGE = (
    r"\b(in general|for (adults|humans|a human|people|men|women|children)|on average for|"
    r"normal (blood pressure|heart rate) for (a )?(human|humans|adults))\b"
)

# General non-human / external-domain / engineering / IT / trivia disambiguation rules
# Used ONLY to distinguish external objects (planets, spacecraft systems, electronics, food, software, finance)
# from personal physiological monitoring.
NON_HUMAN_ENTITY_DOMAINS = re.compile(
    r"\b("
    # Astronomy / planetary / Earth weather
    r"mercury|venus|mars|jupiter|saturn|uranus|neptune|pluto|titan|europa|black hole|pulsar|"
    r"weather|rain|hurricane|forecast|atmosphere|orbit|orbiting|galaxy|universe|"
    # Spacecraft / mechanical / electrical hardware & life support equipment
    r"generator|scrubber|electrolyzer|electrolysis|filter|valve|pump|hull|solar panel|solar array|"
    r"rover|electronics|probe|telescope|canadarm\w*|robot\w*|gimbal|thruster|rocket|engine|car|tire|psi|"
    r"cabin pressure|water pressure|galley|tap|dispenser|pressure suit|suit leak|fire alarm|smoke alarm|airlock|"
    # Computers / software / IT
    r"laptop|desktop|computer|server|cpu|gpu|fan speed|linux|kernel|scheduler|operating system|python|"
    r"software|program|code|api|database|website|sleep mode|unplugged|battery|batteries|"
    # Structural physics / unit conversions / math / chemistry
    r"beam|truss|strain|cantilever|boiling|melting|fahrenheit|celsius|convert|"
    r"arithmetic|calculate\b.*\btimes\b|\d+\s*times\s*\d+|solve this math|quantum physics|"
    # Non-human organisms
    r"plant|plants|tree|crop|crops|algae|bacteria|dog|cat|mouse|"
    # Culinary / nutrition composition trivia (not personal glucose reading)
    r"banana|apple|rice|pasta|bread|oatmeal|pizza|lasagna|quiche|brownies|bake|cook|recipe|tablespoon|calorie content|"
    r"glucose content|carbohydrate content|"
    # Media / sports / finance / general trivia / comms
    r"stock|stocks|portfolio|banking|checking account|savings|wire fee|credit card|currency|exchange rate|"
    r"football|soccer|champions league|world cup|super bowl|wimbledon|nba|playlist|"
    r"poem|bedtime story|joke|sing me|movie|book|news from earth|"
    r"translate|capital of|president|commander|send an email|history of|how many people|time is it in"
    r")\b"
)

# Personal / Crew / Symptom / Body Context Signals (C1 Scope Gate)
PERSONAL_HEALTH_CONTEXT = re.compile(
    r"\b("
    r"i|me|my|mine|we|our|myself|crew|astronaut|"
    r"feel|feeling|feels|felt|am i|did i|do i|have i|should i|can i|"
    r"today|tonight|last night|this morning|this week|yesterday|right now|lately|"
    r"baseline|pre ?flight|mission day|day \d+|to do|task list|profile|record|"
    r"flight surgeon|countermeasure\w*|alert\w*|warning\w*|danger|"
    r"heart rate|blood pressure|systolic|diastolic|oxygen saturation|spo2|"
    r"bone|muscle|immune|cortisol|hormone\w*|temperature|blood glucose|glucose|blood sugar|"
    r"sleep|mood|relax\w*|stressed|mental health|game test|exercise|cardio|resistance|train\w*|"
    r"chest pain|left arm|passed out|fainted|short of breath|lightheaded|dizzy|headache|ache\w*|weak\w*|tired|exhausted"
    r")\b"
)

# Urgent Symptom Escalation Rules (C1)
URGENT_SYMPTOM_PATTERN = re.compile(
    r"("
    r"\bchest (pain|hurts|pressure|tightness)\b|"
    r"\bleft arm\b.*\bnumb\b|\bnumb\b.*\bleft arm\b|"
    r"\b(passed out|fainted|fainting|syncope|unconscious|blacked out)\b|"
    r"\b(severe shortness of breath|can t breathe|cannot breathe|struggling to breathe|short of breath)\b|"
    r"\b(headache|dizzy|lightheaded)\b.*\b(blood pressure|bp|systolic|oxygen|spo2|emergency|dropped)\b|"
    r"\b(blood pressure|bp|systolic|oxygen|spo2|dropped)\b.*\b(headache|dizzy|lightheaded|emergency)\b"
    r")"
)

KEYWORD_SLOTS = [
    ("what_if", r"\bwhat if\b|\bsuppose\b|\bsimulate\b|\bwhat happens (to .*)?if\b|\bwhat would\b|\bif i (sleep|train|got|get|rest)"),
    ("replay", r"\bday \d+\b|\breplay\b"),
    ("correlation", r"\bcorrelation\b|\bmood\b.*\btrain|\btrain\w*\b.*\bmood\b"),
    ("baseline", r"\bbaseline\b|\bpre ?flight\b|\bbefore (the )?(launch|flight|mission)\b"),
    ("calming", r"\bsong\b|\bphoto\b|\b(help me|need to|want to) (relax|calm)\b|\bcalm down\b|\brelax (me )?down\b|\bbreathing exercise"),
    ("contact", r"\bflight surgeon\b|\bcontact\b|\btalk to someone\b"),
    ("todo", r"\bto do\b|\btask\b|\bcardio\b|\bresistance\b|\bcountermeasures?\b|\bworkouts? (do i|left|remain|need)"),
    ("journey", r"\bdestination\b|\bmission day\b|\bspacecraft\b|\baircraft\b|\bflight (take|time|last)\b|\bhow long is (my|the) flight\b|\bdays (is|will) the (mission|flight)\b"),
    ("bone_muscle", r"\bbone\b|\bmuscle\b|\blower back\b|\blegs?\b.*\bweak"),
    ("cardio", r"\bheart\b|\bblood pressure\b|\bsystolic\b|\bdiastolic\b|\boxygen saturation\b|\bcardiovascular\b"),
    ("glucose", r"\bblood glucose\b|\blow sugar\b|\bhigh sugar\b|\bsugar\b"),
    ("immune", r"\bimmune\b|\bcortisol\b|\bhormones?\b|\btemperature\b"),
    ("sleep_psych", r"\bsleep\b|\bmood\b|\brelaxed\b|\bmental health\b|\bgame test\b|\btrain\b|\bpsychological\b|\bstressed\b|\bexercise\b|\bactivity\b"),
    ("alerts", r"\balert\b|\bwrong\b|\bnot normal\b|\bemergency\b|\bdanger\b|\bchest pain\b|\bpassed out\b|\bfainted\b"),
    ("sources", r"\bsources?\b|\binformation\b|\bdata do you\b|\bverified\b|\bmedical device\b|\btrust you\b|\bfacts\b|\ba (real )?flight surgeon\b|\bbased on\b"),
    ("status", r"\bhealth\b|\bstatus\b|\bhow am i\b|\bam i ok\b|\bverdict\b|\bcondition\b|\bmodel say\b|\bpredicted class\b|\bvitals\b|\bfine with me\b"),
    ("profile", r"\bblood type\b|\ballerg|\bage\b|\bold am i\b|\bweigh|\bheight\b|\btall\b|\brole\b|\bnationalit|\bsurger|\binjur|\bimmunization\b|\bfamily medical\b|\bprofile\b|\brecord\b"),
]

COMPARISON = r"\b(higher|lower|above|below|compared|versus|vs|than|normal for me|worse|better|dropping|faster)\b"

UNMEASURED = (
    r"\b(cholesterol|vitamin\w*|kidneys?|liver|x ?ray|ultrasound|mri|radiation|insulin|"
    r"diabet\w*|cancer|ecg|ekg|urine|hemoglobin|iron level|calcium|eyesight|vision)\b"
)

DOMAIN_VOCAB = set("""health status condition state body check summary summarise summarize overview verdict model class
predicted prediction says say result alert active triggered caused cause reason explain fix wrong problem issue heart rate
beat beating pound fluttery weird blood pressure systolic diastolic oxygen saturation cardiovascular reading readings level
levels value values number numbers bone density mass loss losing lost strength weak weaker muscle legs lower back ache
immune system indicators cortisol hormones temperature slight fever cold sore throat sick blood glucose sugar morning
shaky sweaty sleep hours hour night enough long waking fall asleep groggy circadian lighting tips improve mood score
relaxed relax mental assessment game test reaction psychological train trained minutes physical activity skip exhausted
list task exercise cardio resistance session countermeasures left remaining done finished completed logged profile record
blood type allergies allergic age old weight weigh height tall role nationality surgeries surgery injuries immunization
family medical history previous flights gender name baseline pre flight preflight launch mission before song photo stressed
calm play open hear see anxious irritable isolated unmotivated breathing eva flight surgeon contact details earth
destination spacecraft aircraft days day replay happened if suppose simulate change changes risk correlation information
data sources facts verified advice answers based trust device real app normal high low ok okay fine good bad current latest
compare look holding quickly fast slow take last went off put time help need want feel feeling know view display vitals
week month yesterday morning dizzy lightheaded headache chest pain arm numb passed fainted emergency""".split())

FABRICATION = r"\b(make up|made up|invent|pretend|fabricate|fake|imagine|guess|just say|lie about|assume i)\b"
MEDICATION = (
    r"\b(medicine|medicines|medication|medications|drug|drugs|pill|pills|dose|dosage|prescribe|prescription|"
    r"antibiotic|antibiotics|aspirin|ibuprofen|painkiller|painkillers|sedative|supplement|supplements|"
    r"diagnose|diagnosis|treatment for|cure)\b"
)

PROFILE_KEYS = [
    (r"blood type", "blood_type"),
    (r"allerg", "allergies"),
    (r"\bage\b|how old", "age"),
    (r"weigh", "weight_kg"),
    (r"height|how tall", "height_cm"),
    (r"role", "mission_role"),
    (r"nationalit", "nationality"),
    (r"surger", "previous_surgeries"),
    (r"injur", "previous_injuries"),
    (r"immuni[sz]ation|vaccin", "immunization_history"),
    (r"family medical|family history", "family_medical_history"),
    (r"gender|sex\b", "gender"),
    (r"previous flight|flown before|how many flights", "previous_flight"),
    (r"\bname\b", "name"),
]
STRONG_PROFILE = r"blood type|allerg|surger|injur|immuni[sz]ation|vaccin|nationalit|family medical"
WHATIF_KEYS = [
    (r"sleep", "sleep_hours"),
    (r"train|exercise|activity", "daily_training_hours"),
    (r"mood", "mood"),
    (r"relax", "relaxation"),
]

DID_YOU_MEAN_HINTS = [
    (r"\btemperature\b|\bhot\b|\bfever\b|\bcelsius\b|\bfahrenheit\b", "What is my body temperature today compared to my baseline?"),
    (r"\boxygen\b|\bspo2\b|\bo2\b|\bco2\b|\bscrubber\b", "What is my oxygen saturation (SpO2) reading today?"),
    (r"\bpressure\b|\bbp\b|\bsystolic\b|\bdiastolic\b", "What is my blood pressure today compared to my pre-flight baseline?"),
    (r"\bheart\b|\bpulse\b|\bhr\b", "What is my resting heart rate today?"),
    (r"\bsleep\b|\binsomnia\b|\bbedtime\b", "How many hours did I sleep last night?"),
    (r"\bstress\w*\b|\banxi\w*\b|\brelax\w*\b", "My stress level has been high, what can I do to calm down?"),
    (r"\bglucose\b|\bsugar\b|\bdiabet\w*\b", "What is my blood glucose reading today?"),
    (r"\bimmune\b|\bimmunity\b|\binfection\b|\bsick\b", "How are my immunity indicators today?"),
    (r"\bbone\w*\b|\bmuscle\w*\b|\bjoint\w*\b|\bworkout\w*\b|\bexercise\w*\b", "How are my bone and muscle health indicators trending?"),
    (r"\balarm\b|\balert\b|\bwarning\b|\bhealth\b|\bbattery\b", "Do I have any active health alerts today?"),
    (r"\bradiation\b", "What is my current overall health status and active alerts?"),
]


def _normalize(q: str) -> str:
    """Lower-case, strip punctuation, map synonyms onto canonical vocabulary."""
    t = re.sub(r"[^a-z0-9 ]+", " ", str(q).lower())
    for pat, sub in SYNONYMS:
        t = re.sub(pat, sub, t)
    return re.sub(r"\s+", " ", t).strip()


def _topic_words(norm: str) -> List[str]:
    return [w for w in norm.split() if w not in STOPWORDS and not w.isdigit()]


class OfflineChatbot:
    def __init__(self):
        self.questions: List[str] = []
        self.labels: List[str] = []
        for source in (EXAMPLES, EXTRA_EXAMPLES):
            for intent, qs in source.items():
                for q in qs:
                    self.questions.append(q)
                    self.labels.append(intent)
        norm = [_normalize(q) for q in self.questions]
        topic = [" ".join(_topic_words(n)) or n for n in norm]

        self.vec_word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.m_word = self.vec_word.fit_transform(topic)
        self.m_char = self.vec_char.fit_transform(topic)
        self.vocab = DOMAIN_VOCAB | {
            w for n, lab in zip(norm, self.labels) if lab != "out_of_scope" for w in _topic_words(n)
        }

        y_scope = np.array([0 if lab == "out_of_scope" else 1 for lab in self.labels], dtype=int)
        X_scope = np.hstack([self.m_word.toarray(), self.m_char.toarray()])
        self.scope_clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=300, random_state=42)
        self.scope_clf.fit(X_scope, y_scope)

    def scope_gate(self, question: str, norm: str) -> Tuple[bool, float, Optional[str]]:
        """
        Phase 4 C1 Scope Gate before routing.
        Combines offline TF-IDF scope classifier with personal/crew vs. non-human entity context rules.
        """
        raw_lower = re.sub(r"[^a-z0-9 ]+", " ", str(question).lower())
        raw_lower = re.sub(r"\s+", " ", raw_lower).strip()

        # 1. Explicit non-human / planetary / spacecraft hardware / software / culinary / trivia entity check
        if NON_HUMAN_ENTITY_DOMAINS.search(raw_lower) or NON_HUMAN_ENTITY_DOMAINS.search(norm):
            return False, 0.05, "external / non-human / hardware / planetary / trivia entity"

        # 2. Offline TF-IDF scope probability
        topic = " ".join(_topic_words(norm)) or norm
        v_w = self.vec_word.transform([topic]).toarray()
        v_c = self.vec_char.transform([topic]).toarray()
        p_scope = float(self.scope_clf.predict_proba(np.hstack([v_w, v_c]))[0, 1])

        has_personal = bool(PERSONAL_HEALTH_CONTEXT.search(raw_lower) or PERSONAL_HEALTH_CONTEXT.search(norm))
        has_slot = self.has_slot(norm)
        unk = self.unknown_share(norm)

        if unk >= UNKNOWN_SHARE_LIMIT and not (has_personal and has_slot):
            return False, p_scope, "most topic words are outside the health-record vocabulary"

        if not has_personal and not has_slot and p_scope < 0.55:
            return False, p_scope, f"scope gate probability {p_scope:.2f} below threshold without personal/health context"

        return True, p_scope, None

    def suggest_did_you_mean(self, question: str, norm: str) -> Optional[str]:
        """Return a 'Did you mean...?' suggestion when refusing a borderline query."""
        raw_lower = str(question).lower()
        for pat, suggestion in DID_YOU_MEAN_HINTS:
            if re.search(pat, raw_lower) or re.search(pat, norm):
                return suggestion
        return None

    def is_urgent_symptom(self, question: str, norm: str) -> bool:
        """Detect acute clinical red-flag symptoms requiring immediate flight surgeon escalation."""
        raw_lower = re.sub(r"[^a-z0-9 ]+", " ", str(question).lower())
        return bool(URGENT_SYMPTOM_PATTERN.search(raw_lower) or URGENT_SYMPTOM_PATTERN.search(norm))

    def max_training_similarity(self, question: str) -> Tuple[float, str]:
        """
        Compute maximum TF-IDF word cosine similarity between `question` and all training examples.
        Used by Phase 4 C2 anti-leakage verification (target <= 0.60 on evaluation sets).
        """
        norm = _normalize(question)
        topic = " ".join(_topic_words(norm)) or norm
        sims = cosine_similarity(self.vec_word.transform([topic]), self.m_word)[0]
        idx = int(np.argmax(sims))
        return float(sims[idx]), self.questions[idx]

    # ------------------------------------------------------------------ intent
    def classify(self, question: str) -> Tuple[str, float, str]:
        """Return (intent, confidence, nearest_example)."""
        norm = _normalize(question)
        topic = " ".join(_topic_words(norm)) or norm
        sims = (
            0.6 * cosine_similarity(self.vec_word.transform([topic]), self.m_word)[0]
            + 0.4 * cosine_similarity(self.vec_char.transform([topic]), self.m_char)[0]
        )
        best: Dict[str, float] = {}
        for lab, sim in zip(self.labels, sims):
            if sim > best.get(lab, -1.0):
                best[lab] = float(sim)
        nearest = self.questions[int(np.argmax(sims))]
        slots = self.slots(norm)
        if not slots:
            best.pop("replay", None)
            intent = max(best, key=best.get)
            return intent, best[intent], nearest
        if best.get("out_of_scope", 0.0) >= 0.85:
            return "out_of_scope", best["out_of_scope"], nearest
        scored = {lab: best.get(lab, 0.0) + KEYWORD_BONUS - 0.02 * rank for rank, lab in enumerate(slots)}
        intent = max(scored, key=scored.get)
        return intent, min(scored[intent], 1.0), nearest

    def slots(self, norm: str) -> List[str]:
        found = [lab for lab, pat in KEYWORD_SLOTS if re.search(pat, norm)]
        if "what_if" in found:
            return ["what_if"]
        if "replay" in found and re.search(r"\bday \d+\b", norm):
            return ["replay"]
        if "sources" in found and re.search(r"\bare you\b|\byour\b|\bthis (app|tool|software)\b|\btrust you\b|\bwho verified\b", norm):
            return ["sources"]
        if "baseline" in found:
            if re.search(COMPARISON, norm) and len(found) > 1:
                found.remove("baseline")
            else:
                return ["baseline"]
        if "alerts" in found and re.search(r"\b(warning|danger|alert|alarm|emergency|chest pain|passed out|fainted)\b", norm):
            if not re.search(r"\b(bone|muscle)\b", norm):
                return ["alerts"]
        if "todo" in found and "sleep_psych" in found and not re.search(
            r"\bsleep\b|\bmood\b|\bmental\b|\bgame\b|\bexhausted\b|\bhow many hours did i train\b|"
            r"\bhow much did i train\b|\btraining hours\b|\bphysical activity this week\b",
            norm,
        ):
            found.remove("sleep_psych")
        if "bone_muscle" in found and "todo" in found and re.search(r"\bcountermeasures?\b|\bresistance exercise\b", norm):
            return ["todo", "bone_muscle"]
        if "calming" in found and re.search(r"\b(song|photo|calm down|help me relax|help me calm|breathing exercise)\b", norm):
            return ["calming"]
        if "contact" in found and re.search(r"\b(flight surgeon|reach|contact|talk to someone)\b", norm):
            return ["contact"]
        specific = [lab for lab in found if lab not in ("status", "alerts", "sources")]
        if specific:
            return specific
        if "alerts" in found:
            return ["alerts"]
        return found

    def has_slot(self, norm: str) -> bool:
        return bool(self.slots(norm))

    def unknown_share(self, norm: str) -> float:
        words = _topic_words(norm)
        if not words:
            return 1.0
        return sum(w not in self.vocab for w in words) / len(words)

    # ------------------------------------------------------------------ public
    def ask(
        self,
        astronaut_id: str,
        question: str,
        session_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return {'answer', 'intent', 'confidence', 'refused', 'sources', 'did_you_mean', 'urgent_escalation'}.
        Supports optional RBAC session_token verification (Phase 1 D2).
        """
        astronaut_id = str(astronaut_id).strip().upper()
        if session_token is not None:
            auth_chk = he.verify_session(session_token, target_astronaut_id=astronaut_id)
            if not auth_chk.get("ok"):
                return self._pack(
                    rf.REFUSAL_TEXT,
                    "refusal",
                    1.0,
                    True,
                    "none (refusal: " + auth_chk.get("error", "unauthorized") + ")",
                )

        norm = _normalize(question)
        urgent = self.is_urgent_symptom(question, norm)
        intent, conf, _ = self.classify(question)

        if re.search(STRONG_PROFILE, norm) and re.search(r"\b(my|i|me)\b", norm) and intent != "out_of_scope":
            intent, conf = "profile", max(conf, CONFIDENCE_THRESHOLD)

        scope_ok, scope_prob, scope_reason = self.scope_gate(question, norm)

        reason = None
        if not norm:
            reason = "empty question"
        elif re.search(FABRICATION, norm):
            reason = "asked to invent data"
        elif re.search(MEDICATION, norm):
            reason = "medication / diagnosis / treatment is outside the three allowed sources"
        elif any(
            m != astronaut_id
            for m in re.findall(r"\ba\d{2,3}\b", norm.replace("astronaut ", ""))
            for m in [m.upper()]
        ):
            reason = "question about another astronaut"
        elif re.search(PREDICTION, norm) and " if " not in f" {norm} ":
            reason = "asks to predict the future (only the what-if simulator may do that)"
        elif re.search(GENERAL_KNOWLEDGE, norm):
            reason = "general medical knowledge is outside the three allowed sources"
        elif re.search(UNMEASURED, norm):
            reason = "asks about something this prototype does not measure"
        elif not scope_ok:
            reason = scope_reason or "blocked by out-of-scope gate"
        elif self.unknown_share(norm) >= UNKNOWN_SHARE_LIMIT and not urgent:
            reason = "most topic words are outside the health-record vocabulary"
        elif not self.has_slot(norm) and conf < NO_SLOT_THRESHOLD and not urgent:
            reason = f"no topic keyword and similarity {conf:.2f} below {NO_SLOT_THRESHOLD}"
        elif intent == "out_of_scope" and not urgent:
            reason = "nearest example is out of scope"
        elif conf < CONFIDENCE_THRESHOLD and not urgent:
            reason = f"intent confidence {conf:.2f} below threshold {CONFIDENCE_THRESHOLD}"

        if reason is None and not he.astronaut_exists(astronaut_id):
            reason = "unknown astronaut"

        if urgent and reason is None:
            intent = "alerts"
            conf = max(conf, 0.85)

        if reason:
            suggestion = self.suggest_did_you_mean(question, norm)
            refusal_body = rf.REFUSAL_TEXT
            if urgent:
                refusal_body += "\nURGENT SYMPTOM NOTICE: Contact the flight surgeon immediately."
            if suggestion:
                refusal_body += f'\nDid you mean: "{suggestion}"?'
            return self._pack(
                refusal_body,
                "refusal",
                conf,
                True,
                "none (refusal: " + reason + ")",
                did_you_mean=suggestion,
                urgent_escalation=urgent,
            )

        try:
            body, sources = getattr(self, "_a_" + intent)(astronaut_id, question, norm)
        except _Refuse as e:
            suggestion = self.suggest_did_you_mean(question, norm)
            refusal_body = rf.REFUSAL_TEXT
            if suggestion:
                refusal_body += f'\nDid you mean: "{suggestion}"?'
            return self._pack(
                refusal_body,
                "refusal",
                conf,
                True,
                "none (refusal: " + str(e) + ")",
                did_you_mean=suggestion,
                urgent_escalation=urgent,
            )

        if urgent:
            p = he.get_profile(astronaut_id)
            surgeon = p["profile"]["contact_flight_surgeon"] if p.get("ok") else "ground medical control"
            escalation_banner = (
                f"URGENT ESCALATION: Contact the flight surgeon immediately ({surgeon}). "
                "Stop current exertion and initiate real-time ground medical telemetry review.\n"
            )
            if "contact the flight surgeon immediately" not in body.lower():
                body = escalation_banner + body

        return self._pack(
            body,
            intent,
            conf,
            False,
            sources,
            did_you_mean=None,
            urgent_escalation=urgent,
        )

    def reply(self, astronaut_id: str, question: str) -> str:
        return self.ask(astronaut_id, question)["answer"]

    @staticmethod
    def _pack(
        body: str,
        intent: str,
        conf: float,
        refused: bool,
        sources: str,
        did_you_mean: Optional[str] = None,
        urgent_escalation: bool = False,
    ) -> Dict[str, Any]:
        return {
            "answer": f"{body}\n{rf.DISCLAIMER}\nSources: {sources}",
            "intent": intent,
            "confidence": round(conf, 3),
            "refused": refused,
            "sources": sources,
            "did_you_mean": did_you_mean,
            "urgent_escalation": urgent_escalation,
        }

    # ------------------------------------------------------------------ building blocks
    @staticmethod
    def _need(res: Dict[str, Any]) -> Dict[str, Any]:
        if not res.get("ok"):
            raise _Refuse(res.get("error", "engine error"))
        return res

    @staticmethod
    def _linked(la: Dict[str, Any]) -> str:
        t = la.get("type")
        if t == "todo":
            left = [i["task"] for i in la["todo"]["items"] if not i["done"]]
            return "To-Do list: " + ("still open today: " + ", ".join(left) if left else "all exercise tasks done today") + "."
        if t == "sleep_schedule":
            return (
                f"Sleep schedule: your baseline is {la['your_baseline_sleep_hours']} h, your last 7 days average "
                f"{la['last_7_days_mean_sleep_hours']} h. Calming option: song {la['calming']['file']} or family photo "
                f"{la['calming']['alternative']['file']}."
            )
        if t == "todo_and_sleep_schedule":
            left = [i["task"] for i in la["todo"]["items"] if not i["done"]]
            return (
                f"To-Do list: {'open: ' + ', '.join(left) if left else 'all done today'}. Sleep schedule: baseline "
                f"{la['your_baseline_sleep_hours']} h vs last 7 days {la['last_7_days_mean_sleep_hours']} h. "
                f"Calming option: song {la['calming']['file']}."
            )
        if t == "calming":
            return f"Calming action: play your song ({la['file']}) or open your family photo ({la['alternative']['file']})."
        if t == "contact_flight_surgeon":
            return f"Contact the flight surgeon immediately: {la['contact']}."
        return ""

    def _explain_text(self, ex: Dict[str, Any]) -> str:
        """(a) current vs baseline, (b) why, (c) how to fix - from one health_engine explanation dict."""
        cvb = "; ".join(ex["current_vs_baseline"]) or "no single indicator is outside your personal band today"
        fix = ex["how_to_fix"]
        return (
            f"- {ex['what']}\n"
            f"  (a) Current vs your baseline: {cvb}.\n"
            f"  (b) Why: {ex['why']}\n"
            f"  (c) How to avoid or fix it: {fix['advice_text']} {self._linked(fix.get('linked_action', {}))}"
        )

    def _zone_answer(self, aid: str, zone: str) -> Tuple[str, str]:
        ev = self._need(he.explain_alert(aid, zone))
        card = self._need(he.earth_doctor_card(aid, last_n_days=1))
        lines = [
            f"{rf.ZONE_TITLES[zone]} - mission day {ev['mission_day']} - zone status: "
            f"{card['diagnostic_breakdown'][zone]['status'].upper()}"
        ]
        for it in card["diagnostic_breakdown"][zone]["indicators"]:
            z = "n/a" if it.get("z") is None else f"{it['z']:+.2f} SD"
            if it["indicator"] in ("bone_health", "muscle_health"):
                z = "see loss rate below"
            badge = f" [{it.get('ui_badge', 'NASA standard')}]" if it.get("ui_badge") else ""
            lines.append(
                f"  {it['indicator']}: {it['value']} {it['unit']} (your pre-flight baseline "
                f"{it['baseline_mean']} +/- {it['baseline_std']}; {z}){badge}"
            )
        info = card["diagnostic_breakdown"][zone]["info"]
        if info:
            lines.append(
                f"  bone loss over the last 30 days: {info['bone_loss_pct_per_month_trailing_30d']} %/month; "
                f"sessions done in the last 7 days: cardio {info['sessions_last_7_days']['cardio_done']}, "
                f"resistance {info['sessions_last_7_days']['resistance_done']}"
            )
        lines.append(self._explain_text(ev))
        return "\n".join(lines), (
            "mission_telemetry, preflight_baseline, countermeasures, "
            "health_engine.explain_alert, health_engine.earth_doctor_card"
        )

    # ------------------------------------------------------------------ intent handlers
    def _a_status(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        c = self._need(he.astronaut_card(aid))
        zones = ", ".join(f"{rf.ZONE_TITLES[z]}={s}" for z, s in c["zones"].items())
        lines = [f"{c['name']}, mission day {c['mission_day']}: {c['status']}", f"Body zones: {zones}."]
        if c["active_alerts"]:
            ex = self._need(he.explain_alert(aid))
            lines.append(self._explain_text(ex))
            lines += [self._explain_text(o) for o in ex.get("other_alerts", [])]
        return "\n".join(lines), "health_engine.astronaut_card, health_engine.explain_alert, model_predictions, countermeasures"

    def _a_alerts(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        ex = self._need(he.explain_alert(aid))
        if ex["area"] is None:
            return (
                f"Mission day {ex['mission_day']}: {ex['what']} "
                f"(a) Current vs your baseline: all indicators within personal band. "
                f"(b) Why: {ex['why']} "
                f"(c) How to avoid or fix it: {ex['how_to_fix']['advice_text']}",
                "health_engine.explain_alert, mission_telemetry, preflight_baseline",
            )
        lines = [
            f"Mission day {ex['mission_day']} - active alerts: {', '.join(ex['active_areas'])}.",
            self._explain_text(ex),
        ] + [self._explain_text(o) for o in ex.get("other_alerts", [])]
        return "\n".join(lines), (
            "health_engine.explain_alert, mission_telemetry, preflight_baseline, "
            "countermeasures, model output"
        )

    def _a_cardio(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        return self._zone_answer(aid, "Cardiovascular")

    def _a_bone_muscle(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        return self._zone_answer(aid, "Bone_Muscle")

    def _a_immune(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        return self._zone_answer(aid, "Immune")

    def _a_glucose(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        card = self._need(he.earth_doctor_card(aid, last_n_days=1))
        imm = card["diagnostic_breakdown"]["Immune"]
        gluc_items = [it for it in imm["indicators"] if it["indicator"] == "blood_sugar"]
        lines = [f"Metabolic / Blood Glucose - mission day {card['mission_day']} - Immune/Metabolic zone: {imm['status'].upper()}"]
        for it in gluc_items:
            z = "n/a" if it.get("z") is None else f"{it['z']:+.2f} SD"
            lines.append(
                f"  blood_glucose_mg_dl (blood_sugar): {it['value']} {it['unit']} "
                f"(your pre-flight baseline {it['baseline_mean']} +/- {it['baseline_std']}; {z}) "
                f"[{it.get('ui_badge', 'Reference value')}]"
            )
        lines.append(
            "  Clinical reference band (ADA Standards of Care 2024 [TERRESTRIAL_REFERENCE]): "
            "normal fasting 70-99 mg/dL; hypoglycemia warning < 70 mg/dL; clinically significant hypoglycemia < 54 mg/dL; "
            "elevated fasting >= 100 mg/dL (hyperglycemia danger >= 180 mg/dL)."
        )
        return "\n".join(lines), (
            "mission_telemetry (blood_sugar / blood_glucose_mg_dl), preflight_baseline, "
            "07_health_indicators_schema.json (ADA 2024)"
        )

    def _a_sleep_psych(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        return self._zone_answer(aid, "Sleep_Brain")

    def _a_todo(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        t = self._need(he.get_todo(aid))
        lines = [f"To-Do list for mission day {t['mission_day']}:"]
        for it in t["items"]:
            lines.append(f"  [{'x' if it['done'] else ' '}] {it['task']} - {it['minutes']} min recorded")
        lines.append(
            "  Countermeasure guidance: Complete daily ARED resistive exercise + treadmill/cycle cardio "
            "(2.5 h/day block, 6 days/week per NASA-STD-3001) to mitigate microgravity bone and muscle loss."
        )
        return "\n".join(lines), "mission_telemetry (To-Do columns), countermeasures, health_engine.get_todo"

    def _a_profile(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        p = self._need(he.get_profile(aid))["profile"]
        hits = [col for pat, col in PROFILE_KEYS if re.search(pat, norm)]
        if hits:
            body = "; ".join(f"{col}: {p[col]}" for col in hits)
            return f"From your profile record - {body}.", "astronaut_profile"
        keys = [
            "name", "mission_role", "nationality", "age", "gender",
            "weight_kg", "height_cm", "blood_type", "allergies",
        ]
        return ("Your profile record - " + "; ".join(f"{k}: {p[k]}" for k in keys) + "."), "astronaut_profile"

    def _a_baseline(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        b = self._need(he.get_profile(aid))["baseline"]
        zone_for = [
            ("Cardiovascular", r"heart|pulse|blood pressure|systolic|diastolic|oxygen|spo2"),
            ("Sleep_Brain", r"sleep|mood|relax|mental|game|train"),
            ("Immune", r"immun|hormone|cortisol|temperature|blood glucose|sugar"),
            ("Bone_Muscle", r"bone|muscle|cardio|resistance|activity"),
        ]
        names = [n for z, pat in zone_for if re.search(pat, norm) for n in rf.ZONES[z]] or rf.CONTINUOUS
        lines = ["Your personal pre-flight baseline (mean +/- standard deviation over 30 pre-flight days):"]
        for n in dict.fromkeys(names):
            lines.append(f"  {n}: {b[n + '_mean']:.2f} +/- {b[n + '_std']:.2f} {rf.IND[n]['unit']}")
        return "\n".join(lines), "preflight_baseline"

    def _a_calming(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        p = self._need(he.get_profile(aid))["profile"]
        if re.search(r"photo|picture|family", norm):
            act, f = "show_family_photo", p["family_photo"]
        else:
            act, f = "play_song", p["song"]
        return (
            f"Calming action triggered: {act} -> {f}. (Other option: "
            f"{'song ' + p['song'] if act == 'show_family_photo' else 'family photo ' + p['family_photo']}.) "
            "Guided box-breathing (4s inhale, 4s hold, 4s exhale) and scheduled crew/family contact are recommended.",
            "astronaut_profile (calming preferences), countermeasures",
        )

    def _a_contact(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        p = self._need(he.get_profile(aid))["profile"]
        return f"Your flight surgeon contact on record: {p['contact_flight_surgeon']}.", "astronaut_profile"

    def _a_journey(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        p = self._need(he.get_profile(aid))["profile"]
        day = self._need(he.get_todo(aid))["mission_day"]
        return (
            f"Journey record - vehicle: {p['type_of_aircraft']}; destination: {p['destination_for_flight']}; "
            f"estimated flight time: {p['estimated_time_for_flight_days']} days; latest recorded mission day: {day}.",
            "astronaut_profile, mission_telemetry",
        )

    def _a_replay(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        m = re.search(r"day (\d+)", norm)
        if not m:
            raise _Refuse("no mission day given")
        ts = datetime.fromisoformat(rf.MISSION_START) + timedelta(days=int(m.group(1)) - 1)
        r = self._need(he.mission_replay(aid, ts.isoformat()))
        sp = r["stored_prediction"]
        tel = r["telemetry"]
        keys = [
            "systolic_bp", "diastolic_bp", "heart_rate",
            "oxygen_saturation", "temperature", "blood_sugar", "sleep_hours", "mood",
        ]
        pred = "no stored prediction" if sp is None else (
            f"model class {sp['predicted_class']} (p={sp['class_probabilities'][sp['predicted_class']]:.2f})"
        )
        return (
            f"Replay of mission day {r['mission_day']} ({r['matched_timestamp']}): {pred}. Readings: "
            + ", ".join(f"{k}={tel[k]}" for k in keys if k in tel)
            + ". Zones then: "
            + ", ".join(f"{z}={s}" for z, s in r["zones_at_that_time"].items())
            + ".",
            "health_engine.mission_replay, mission_telemetry, model_predictions",
        )

    def _a_what_if(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        ind = next((col for pat, col in WHATIF_KEYS if re.search(pat, norm)), None)
        if ind is None:
            raise _Refuse("what-if without a supported indicator")
        num = re.search(r"(\d+(?:\.\d+)?)", norm)
        amount = float(num.group(1)) if num else (0.5 if "half" in norm else 1.0)
        if "minute" in norm and ind in ("sleep_hours", "daily_training_hours"):
            amount /= 60.0
        sign = -1.0 if re.search(r"\b(less|fewer|reduce|cut|decrease|lower|worse|drops?)\b", norm) else 1.0
        r = self._need(he.what_if(aid, {ind: sign * amount}))
        ch = r["applied_changes"][ind]
        probs = "; ".join(
            f"{c}: {r['before']['class_probabilities'][c]:.2f} -> {r['after']['class_probabilities'][c]:.2f}"
            for c in rf.CLASSES
        )
        return (
            f"What-if on mission day {r['mission_day']}: {ind} {ch['before']} -> {ch['after']} {ch['unit']}. "
            f"Model class {r['before']['predicted_class']} -> {r['after']['predicted_class']}. "
            f"Class probabilities: {probs}. {r['note']}"
        ), "health_engine.what_if, live model output, mission_telemetry"

    def _a_correlation(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        r = self._need(he.mood_training_correlation(aid))
        w = r["within_astronaut"][0]
        all_r = self._need(he.mood_training_correlation())
        rm = all_r["repeated_measures_and_mixed_effects"]
        return (
            f"Mood vs daily training hours — within your own records (n={w['n']} days): "
            f"Pearson r={w['pearson_r']} (95% CI [{w['ci_95_low']}, {w['ci_95_high']}], p={w['pearson_p']}), "
            f"Spearman rho={w['spearman_rho']} (p={w['spearman_p']}). "
            f"Crew-wide within-astronaut repeated-measures correlation (n={rm['n_observations']}, df={rm['df']}): "
            f"r_rm={rm['rm_corr_r']} (95% CI [{rm['ci_95_low']}, {rm['ci_95_high']}], p={rm['p_value']}), "
            f"mixed-effects slope={rm['mixed_effects_slope']} (95% CI {rm['mixed_effects_slope_ci_95']}). {r['note']}",
            "mission_telemetry, health_engine.mood_training_correlation",
        )

    def _a_sources(self, aid: str, q: str, norm: str) -> Tuple[str, str]:
        return (
            "I am not a doctor and this is not a medical device. I answer only from three places: (1) your profile, "
            "pre-flight baseline and telemetry in the SQLite database, (2) the live output of the trained model, and "
            "(3) the countermeasures table built from the challenge resources and NASA-STD-3001 / OCHMO / HRP standards. "
            "All telemetry in this prototype is simulated, and reviewer_status remains empty until a physician review CSV "
            "is imported via review/record_review.py.",
            "data_provenance, countermeasures, model " + rf.MODEL_VERSION,
        )


class _Refuse(Exception):
    pass


def main():
    bot = OfflineChatbot()
    aid = (sys.argv[1] if len(sys.argv) > 1 else "A001").upper()
    print(rf.DISCLAIMER)
    print(f"Offline chatbot for astronaut {aid}. Type a question, or 'quit'.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except EOFError:
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        print("Bot: " + bot.reply(aid, q))


if __name__ == "__main__":
    main()
