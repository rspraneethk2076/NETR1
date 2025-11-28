# frame_processor.py
import time
from datetime import datetime

import state
from llm_utils import llava_scene_description, llama_generate_object_action_json
from face_utils import get_face_embedding, identify_person
from video_utils import detect_changes
from graph_utils import update_scene_graph, save_scene_graph_image
from logging_utils import save_json_log


class FrameProcessor:
    """
    Processes frames at a fixed interval (e.g., every 3s).
    Called from a background worker (NOT directly in the video callback).
    """

    def __init__(self, interval: float = 3.0):
        self.interval = interval
        self.last_time = 0.0
        self.previous_frame = None
        self.previous_face_emb = None

    def process(self, frame_bgr):
        """
        frame_bgr: numpy array (H, W, 3) in BGR format
        """
        now = time.time()
        if now - self.last_time < self.interval:
            # Too soon since last processing; skip this frame
            return
        self.last_time = now

        timestamp = datetime.now().strftime("%H:%M:%S")

        # 1) LLaVA description
        description_text = llava_scene_description(frame_bgr)

        # 2) LLaMA structured object-actions JSON
        object_actions = llama_generate_object_action_json(description_text)

        # 3) Face embedding + person ID
        face_emb = get_face_embedding(frame_bgr)
        person_id = identify_person(face_emb)

        # 4) Change detection
        change_info = detect_changes(
            frame_bgr,
            self.previous_frame,
            face_emb,
            self.previous_face_emb,
        )

        # 5) Extract flat lists
        objects = [oa.get("object") for oa in object_actions]
        actions = [oa.get("association") for oa in object_actions]

        # Update previous frame & embedding
        self.previous_frame = frame_bgr.copy()
        self.previous_face_emb = face_emb

        # 6) Build log entry
        entry = {
            "timestamp": timestamp,
            "person_id": person_id,
            "object_actions": object_actions,
            "objects": objects,
            "actions": actions,
            "description": description_text,
            "changes": change_info,
        }

        # 7) Update in-memory log + scene graph (thread-safe)
        with state.lock:
            state.memory.append(entry)
            update_scene_graph(person_id, object_actions, timestamp)
            save_scene_graph_image(person_id, timestamp)

        # 8) Persist JSON log
        save_json_log(entry)

        # Debug in server logs
        print(f"\n📝 FRAME @ {timestamp}")
        print("Raw Description:", description_text)
        print("Generated JSON:", object_actions)
        print("Changes:", change_info)
