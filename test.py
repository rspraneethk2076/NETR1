import cv2
import base64
import time
import requests
import threading
import json
import numpy as np
from datetime import datetime
import insightface
import os
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL MEMORY + JSON LOG STORAGE
# ============================================================
memory = []                    # memory for chat
person_registry = {}           # {person_id : embedding}
next_person_id = 1
lock = threading.Lock()

JSON_LOG_FILE = "video_log.json"

# Folder to store graph images
GRAPH_DIR = "graph_snapshots"
os.makedirs(GRAPH_DIR, exist_ok=True)

# Global scene graph (person ↔️ object)
scene_graph = nx.DiGraph()

# Create json log file if not present
if not os.path.exists(JSON_LOG_FILE):
    with open(JSON_LOG_FILE, "w") as f:
        json.dump([], f, indent=4)

# ============================================================
# FACE MODEL (InsightFace)
# ============================================================
face_model = insightface.app.FaceAnalysis()
face_model.prepare(ctx_id=0, det_size=(320, 320))

previous_face_emb = None
previous_frame = None


# ============================================================
# STREAM PARSER
# ============================================================
def parse_stream(r):
    text = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "response" in obj:
                text += obj["response"]
        except:
            continue
    return text.strip()


# ============================================================
# LLaVA FOR RAW DESCRIPTION
# ============================================================
def llava_scene_description(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return "encode error"

    img64 = base64.b64encode(buf).decode()
    prompt = "Describe in detail what the person is doing in this frame."

    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [img64],
        "stream": True
    }

    r = requests.post("http://localhost:11434/api/generate",
                      json=payload, stream=True)

    return parse_stream(r)


# ============================================================
# JSON STRUCTURER (LLaMA)
# ============================================================
def llama_generate_object_action_json(description_text):
    json_prompt = f"""
You will be given a natural language description of a scene.
Extract ONLY objects associated with the visible person and how they interact.

Return ONLY in JSON format:
{{
  "object_actions": [
    {{
      "object": "<object_name>",
      "association": "<action>"
    }}
  ]
}}

If no objects exist, return:
{{
  "object_actions": []
}}

Description: \"\"\"{description_text}\"\"\"
"""

    payload = {
        "model": "llama3",
        "prompt": json_prompt,
        "stream": True
    }

    r = requests.post("http://localhost:11434/api/generate",
                      json=payload, stream=True)

    raw = parse_stream(r)
    cleaned = extract_json_block(raw)

    if cleaned == "":
        print("⚠️ No valid JSON found in response:", raw)
        return []

    try:
        result = json.loads(cleaned)
        return result.get("object_actions", [])
    except Exception as e:
        print("⚠️ JSON Parse Error:", e)
        print("Raw cleaned JSON:", cleaned)
        return []


def extract_json_block(text):
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start:end + 1].strip()


# ============================================================
# FACE EMBEDDING + PERSON ID
# ============================================================
def get_face_embedding(frame):
    faces = face_model.get(frame)
    if len(faces) == 0:
        return None
    return faces[0].embedding


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def identify_person(face_emb):
    global next_person_id

    if face_emb is None:
        return None

    best_id = None
    best_sim = 0

    for pid, emb in person_registry.items():
        sim = cosine_sim(face_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_id = pid

    if best_sim > 0.60:
        return best_id

    # New person
    person_registry[next_person_id] = face_emb
    next_person_id += 1
    return next_person_id - 1


# ============================================================
# NETWORKX GRAPH UPDATE + SAVE IMAGE
# ============================================================
def update_scene_graph(person_id, object_actions, timestamp):
    if person_id is None:
        return

    person_node = f"person_{person_id}"

    if not scene_graph.has_node(person_node):
        scene_graph.add_node(person_node, type="person", person_id=person_id)

    for oa in object_actions:
        obj_name = oa.get("object")
        relation = oa.get("association")

        if not obj_name:
            continue

        obj_node = f"object_{obj_name}"

        if not scene_graph.has_node(obj_node):
            scene_graph.add_node(obj_node, type="object", name=obj_name)

        scene_graph.add_edge(
            person_node,
            obj_node,
            association=relation,
            last_seen=timestamp
        )


def save_scene_graph_image(person_id, timestamp):
    if scene_graph.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(scene_graph)

    person_nodes = [n for n, d in scene_graph.nodes(data=True) if d.get("type") == "person"]
    object_nodes = [n for n, d in scene_graph.nodes(data=True) if d.get("type") == "object"]

    plt.figure(figsize=(7, 5))

    nx.draw_networkx_nodes(scene_graph, pos, nodelist=person_nodes,
                           node_color="lightblue", node_shape="s", node_size=800)
    nx.draw_networkx_nodes(scene_graph, pos, nodelist=object_nodes,
                           node_color="lightgreen", node_shape="o", node_size=600)

    nx.draw_networkx_edges(scene_graph, pos, arrows=True, arrowstyle="->")

    labels = {}
    for n, d in scene_graph.nodes(data=True):
        labels[n] = d.get("name", n)

    nx.draw_networkx_labels(scene_graph, pos, labels=labels, font_size=8)

    edge_labels = nx.get_edge_attributes(scene_graph, "association")
    nx.draw_networkx_edge_labels(scene_graph, pos, edge_labels=edge_labels, font_size=7)

    plt.axis("off")
    plt.tight_layout()

    safe_ts = timestamp.replace(":", "-")
    pid_str = f"p{person_id}"
    filename = f"graph_{safe_ts}_{pid_str}.png"

    out_path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"📷 Scene graph image saved: {out_path}")


def debug_print_scene_graph():
    print("📊 Current scene graph edges:")
    for u, v, data in scene_graph.edges(data=True):
        assoc = data.get("association", "")
        ts = data.get("last_seen", "")
        print(f"  {u} --({assoc})--> {v} [last_seen: {ts}]")


# ============================================================
# CHANGE DETECTION
# ============================================================
def detect_changes(frame, prev_frame, face_emb, prev_face_emb):
    if prev_frame is None:
        return "First frame — no comparison."

    changes = []

    diff = cv2.absdiff(prev_frame, frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    pct = (np.count_nonzero(diff_gray) / diff_gray.size) * 100

    if pct > 5:
        changes.append(f"Scene changed by {pct:.1f}%.")

    if face_emb is not None and prev_face_emb is not None:
        sim = cosine_sim(face_emb, prev_face_emb)
        if sim > 0.55:
            changes.append("Same person detected.")
        else:
            changes.append("Different person or unclear.")
    else:
        changes.append("No face detected.")

    return " ".join(changes)


# ============================================================
# SAVE JSON LOG (THREAD SAFE)
# ============================================================
def save_json_log(entry):
    with lock:
        try:
            if os.path.exists(JSON_LOG_FILE) and os.path.getsize(JSON_LOG_FILE) > 0:
                with open(JSON_LOG_FILE, "r") as f:
                    data = json.load(f)
            else:
                data = []
        except:
            print("⚠️ video_log.json corrupted — resetting.")
            data = []

        data.append(entry)

        with open(JSON_LOG_FILE, "w") as f:
            json.dump(data, f, indent=4)


# ============================================================
# WORKER THREAD (EACH FRAME)
# ============================================================
def capture_worker(cap, interval=3):
    global previous_frame, previous_face_emb

    while True:
        time.sleep(interval)

        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = datetime.now().strftime("%H:%M:%S")

        description_text = llava_scene_description(frame)
        object_actions = llama_generate_object_action_json(description_text)

        face_emb = get_face_embedding(frame)
        person_id = identify_person(face_emb)

        change_info = detect_changes(frame, previous_frame, face_emb, previous_face_emb)

        objects = [oa["object"] for oa in object_actions]
        actions = [oa["association"] for oa in object_actions]

        previous_frame = frame.copy()
        previous_face_emb = face_emb

        entry = {
            "timestamp": timestamp,
            "person_id": person_id,
            "object_actions": object_actions,
            "objects": objects,
            "actions": actions,
            "description": description_text,
            "changes": change_info
        }

        with lock:
            memory.append(entry)
            update_scene_graph(person_id, object_actions, timestamp)
            save_scene_graph_image(person_id, timestamp)

        save_json_log(entry)

        print(f"\n📝 FRAME @ {timestamp}")
        print("Raw Description:", description_text)
        print("Generated JSON:", object_actions)
        print("Changes:", change_info)
        debug_print_scene_graph()
        print("\n")


# ============================================================
# CHAT WITH MEMORY
# ============================================================
def chat_with_memory(question):
    with lock:
        context = "\n".join([
            f"[{m['timestamp']}] ID={m['person_id']} OBJ_ACT={m['object_actions']}\n"
            f"DESC: {m['description']}\n"
            f"CHANGES: {m['changes']}\n"
            for m in memory
        ])

    prompt = f"""
You are a memory-based video reasoning AI.
Memory entries include timestamp, person_id, object_actions, description, and scene changes.

Memory log:
{context}

User question: {question}

Answer ONLY using the memory.
"""

    payload = {"model": "llama3", "prompt": prompt, "stream": True}
    r = requests.post("http://localhost:11434/api/generate",
                      json=payload, stream=True)

    return parse_stream(r)


# ============================================================
# MAIN APP LOOP
# ============================================================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not available")
        return

    print("🎥 Live feed started")
    print("Press 'c' for chat, 'q' to quit")

    threading.Thread(target=capture_worker, args=(cap,), daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Live Feed", frame)
        k = cv2.waitKey(1)

        if k == ord("q"):
            break

        if k == ord("c"):
            question = input("\n💬 Ask: ")
            print("\n🤖:", chat_with_memory(question))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
