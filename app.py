import streamlit as st
import boto3
import json
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
import re

# --- AWS CONFIG ---
REGION = "us-east-1"
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table("pure-cycle-leaderboard")

st.set_page_config(page_title="Pure-Cycle AI", page_icon="🌊", layout="wide")

# --- UI ---
st.title("🌊 Pure-Cycle: Ghost Fiber Analyzer")
st.markdown("### For SITA Sustainability Hackathon")

with st.expander("Problem statement & Why this matters", expanded=True):
    st.markdown(
        """
        **What's the Problem?**
        Synthetic textiles shed microfibers during washing. A significant share of ocean microplastics comes from textiles, but most people
        have no simple way to understand a garment’s shedding risk from the care label.

        **Why Now?**
        Small behavior changes (cold wash, gentler cycle, fiber capture) can meaningfully reduce microfiber pollution—if guidance is fast,
        practical, and personalized to what the garment is made of.

        **What Pure-Cycle App Does...**
        Upload a care-label photo to estimate “Ocean Impact”, get 2 easy washing tips, and see an estimated microplastics reduction.
        """
    )

# Sidebar Leaderboard
with st.sidebar:
    st.header("Community Stats")
    try:
        # Simple scan counter
        res = table.scan(Select='COUNT')
        st.metric("Total Items Scanned", res['Count'])
    except:
        st.metric("Total Items Scanned", "0")

# Main Interface
uploaded_file = st.file_uploader("Snap a photo of your clothing's care label", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Label for Analysis", width=350)
    
    # Image Preparation
    MAX_EDGE_PX = 1568
    #Anthropic recommends to resize image pixel by pixel to 1568 to avoid token limit issues
    buffered = BytesIO()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    w, h = img.size
    scale = min(1.0, MAX_EDGE_PX / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img.save(buffered, format="WEBP", quality=80, method=6)
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    if st.button("Analyze Ocean Impact"):
        with st.spinner("Claude Haiku is calculating fiber leakage..."):
            
            prompt_text = """
            <instructions>
            You are a Marine Biologist and Textile Scientist. You specialize in microfiber pollution from textiles and evidence-based washing practices that reduce microplastic runoff into aquatic ecosystems.

            Base your recommendations on established environmental research regarding microfiber shedding, including factors such as material type (synthetic vs natural), water temperature, agitation level, and spin speed.
            </instructions>
            <task>
            1. Identify all materials listed on the clothing label and their percentages.

            2. Classify each material by microplastic shedding risk:
               - High Risk: polyester, nylon, acrylic, elastane/spandex, synthetic blends
               - Medium Risk: rayon/viscose and partial synthetic blends
               - Low Risk: cotton, linen, wool

            3. Provide an 'Ocean Impact Score' (1–10), where:
               - 1–3 = low (mostly natural fibers)
               - 4–6 = mixed/blended materials
               - 7–10 = high synthetic content and high shedding potential

            4. Generate 2–3 practical washing recommendations that directly reduce microplastic runoff.
               These must:
               - Be easy to follow in a single wash (no multi-step routines)
               - Include specific machine settings where relevant:
                 • Cold water (≤30°C / 86°F)
                 • Gentle/Delicate cycle
                 • Low to medium spin speed
               - Include at least one microplastic-specific action when synthetics are present (e.g., use of a microfiber bag or filter, washing less frequently)

            5. Provide an estimated reduction in microplastic shedding (as a percentage range) if the recommendations are followed.
               - Base estimates on general research trends (e.g., colder water and reduced agitation can reduce shedding by ~20–50%)
               - Adjust estimates depending on material risk level

            6. Keep tone action-oriented and informative, not judgmental or alarmist.

            7. Output format:
            Return ONLY valid JSON. Do not wrap in markdown fences.

            Schema:
            {
              "materials": "string (all materials + percentages as written)",
              "material_risk": [{"material": "string", "risk": "High|Medium|Low"}],
              "score": 1-10,
              "why_it_matters": "string (1-2 sentences)",
              "recommended_wash_settings": ["string", "string", "string"],
              "microplastic_reduction_actions": ["string", "string"],
              "estimated_impact_reduction": "string (e.g. \"35–50%\")"
            }
            </task>
            """

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": img_b64}},
                        {"type": "text", "text": prompt_text}
                    ]
                }]
            })

            try:
                # API Call
                response = bedrock.invoke_model(
                    modelId="arn:aws:bedrock:us-east-1:048271427261:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0",
                    body=body,
                )
                raw_body = response.get("body").read()
                if not raw_body:
                    raise ValueError("Bedrock returned an empty response body.")

                raw_text = raw_body.decode("utf-8", errors="replace") if isinstance(raw_body, (bytes, bytearray)) else str(raw_body)
                result = json.loads(raw_text)

                model_text = ""
                try:
                    model_text = result["content"][0].get("text", "")
                except Exception:
                    model_text = ""

                cleaned = (model_text or "").strip()
                if cleaned.startswith("```"):
                    # Strip markdown fences like ``` ... ```
                    lines = cleaned.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()

                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    start = cleaned.find("{")
                    end = cleaned.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        data = json.loads(cleaned[start : end + 1])
                    else:
                        st.error("Model returned non-JSON. Showing raw model output below.")
                        st.code(cleaned or model_text or raw_text)
                        raise

                materials = str(data.get("materials", "")).strip()
                score = data.get("score", None)
                why_it_matters = str(data.get("why_it_matters", "")).strip()
                estimated_reduction = str(data.get("estimated_impact_reduction", "")).strip()
                wash_settings = data.get("recommended_wash_settings", []) or []
                actions = data.get("microplastic_reduction_actions", []) or []

                # Save to DynamoDB
                table.put_item(Item={
                    'scan_id': str(datetime.now().timestamp()),
                    'materials': materials or "Unknown",
                    'score': int(score) if isinstance(score, int) or (isinstance(score, str) and str(score).isdigit()) else -1,
                    'estimated_impact_reduction': estimated_reduction,
                    'why_it_matters': why_it_matters,
                    'recommended_wash_settings': wash_settings,
                    'microplastic_reduction_actions': actions,
                    'date': str(datetime.now().date())
                })

                # Display Result
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Ocean Impact Score", f"{score}/10" if score is not None else "—")
                c2.info(f"**Composition:** {materials or '—'}")
                if estimated_reduction:
                    c3.success(f"**Estimated impact reduction:** {estimated_reduction}")
                
                st.subheader("Why It Matters")
                st.write(why_it_matters or "—")

                st.subheader("Recommended Wash Settings")
                if wash_settings:
                    for s in wash_settings:
                        st.write(f"🔹 {s}")
                else:
                    st.write("—")

                st.subheader("Microplastic Reduction Actions")
                if actions:
                    for a in actions:
                        st.write(f"🔹 {a}")
                else:
                    st.write("—")

                with st.expander("Raw JSON", expanded=False):
                    st.code(json.dumps(data, indent=2))

            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")
