"""Demo web server — serves the frontend with mock backend (no LLM needed).

Usage:
    python3 demo_server.py
    # Then open http://localhost:8000 in browser
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path

app = FastAPI(title="Medical Rehab Agent — Demo")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


class PatientInput(BaseModel):
    age: int = Field(..., description="Patient age")
    symptoms: str = Field(..., description="Symptom description")
    history: str = Field(default="无", description="Medical history")


@app.post("/analyze/full")
def analyze_full(patient: PatientInput):
    """Mock endpoint — returns demo data based on symptom patterns."""
    # Import the demo mock logic inline to avoid circular deps
    data = patient.model_dump()
    s = data["symptoms"]

    if "不说话" in s or "社交回避" in s:
        return _selective_mutism_response(data)
    if "重复音节" in s or "口吃" in s or "延长发音" in s:
        return _stuttering_response(data)
    return _dld_response(data)


def _dld_response(data):
    return {
        "patient": data,
        "symptom_analysis": {
            "possible_conditions": [
                {"name": "Developmental Language Disorder (DLD)", "confidence": "高",
                 "matching_symptoms": ["语言表达困难", "发音不清"], "unmatched_symptoms": ["注意力不集中"]},
                {"name": "ADHD", "confidence": "中",
                 "matching_symptoms": ["注意力不集中"], "unmatched_symptoms": ["发音不清", "语言表达困难"]},
            ],
            "key_symptoms": ["发音不清", "语言表达困难", "注意力不集中"],
            "analysis_summary": "患者表现为语言表达和语音方面的障碍，同时伴有注意力问题。主要考虑发育性语言障碍（DLD），需注意是否合并ADHD。"
        },
        "risk_assessment": {
            "risk_level": "中", "risk_score": 5,
            "dimensions": {
                "communication_impact": {"level": "高", "description": "语言表达和发音均受影响，沟通能力明显受限"},
                "social_impact": {"level": "中", "description": "沟通困难可能导致社交回避"},
                "academic_impact": {"level": "中", "description": "注意力不集中影响课堂学习"},
                "emotional_impact": {"level": "低", "description": "暂无明显情绪问题表现"},
            },
            "risk_factors": ["语言功能的广泛性影响", "注意力问题可能加重语言障碍"],
            "protective_factors": ["年龄小神经可塑性强", "无重大疾病史"],
            "urgency": "尽快干预",
            "recommendation": "建议尽快启动语言康复训练，同时评估注意力情况，必要时联合干预。"
        },
        "rehab_plan": {
            "diagnosis_summary": "发育性语言障碍（DLD），可能合并注意缺陷特征",
            "rehab_plan": {
                "overall_goal": "改善语言表达能力和语音清晰度，提升注意力持续水平",
                "duration_estimate": "3-6个月",
                "phases": [
                    {"phase": "初期评估与基础训练", "duration": "2-4周",
                     "objectives": ["完成详细语言评估", "建立训练关系", "基础口腔运动训练"],
                     "activities": [
                         {"name": "口腔运动训练", "description": "增强口腔肌肉协调性", "frequency": "每日10分钟", "method": "吹气球、舌头体操、唇部按摩"},
                         {"name": "听觉辨别训练", "description": "训练区分正确与错误发音", "frequency": "每周3次，每次15分钟", "method": "最小对立体辨别游戏"},
                     ]},
                    {"phase": "核心语言训练", "duration": "6-8周",
                     "objectives": ["扩展词汇量", "改善语音清晰度", "提升语句长度"],
                     "activities": [
                         {"name": "词汇扩展练习", "description": "从名词到动词到形容词逐步推进", "frequency": "每日20分钟", "method": "使用图卡命名、情景描述、词汇接龙游戏"},
                         {"name": "构音训练", "description": "针对错误音素系统矫正", "frequency": "每周2次，每次30分钟", "method": "单音→音节→词语→句子渐进训练"},
                         {"name": "正念注意力训练", "description": "提升注意力和自我调节", "frequency": "每日5-10分钟", "method": "专注呼吸练习、听觉注意游戏"},
                     ]},
                    {"phase": "巩固与泛化", "duration": "4-8周",
                     "objectives": ["在自然情境中运用语言", "维持注意力", "社交沟通提升"],
                     "activities": [
                         {"name": "叙事训练", "description": "提升篇章组织和叙事能力", "frequency": "每周3次，每次20分钟", "method": "使用图片故事卡引导讲述"},
                         {"name": "社交对话练习", "description": "在游戏情境中练习对话", "frequency": "每周2次", "method": "角色扮演、合作桌游、话题讨论"},
                     ]},
                ]
            },
            "home_practice": ["家长每日15分钟慢速清晰对话", "亲子共读鼓励描述图片", "日常场景词汇练习", "定时器辅助注意力任务"],
            "follow_up": {"next_assessment": "4-6周后", "monitoring_indicators": ["语音清晰度", "平均语句长度MLU", "注意力持续时间", "社交参与度"]},
            "precautions": ["避免疲劳状态下强迫训练", "若注意力持续问题应评估ADHD", "家长保持积极鼓励", "保持训练趣味性"]
        },
        "rag_sources": [
            {"id": "KB001", "disease": "Developmental Language Disorder"},
            {"id": "KB003", "disease": "ADHD"},
            {"id": "KB002", "disease": "Speech Sound Disorder"},
        ]
    }


def _selective_mutism_response(data):
    return {
        "patient": data,
        "symptom_analysis": {
            "possible_conditions": [
                {"name": "Selective Mutism", "confidence": "高",
                 "matching_symptoms": ["特定场合不说话", "社交回避", "焦虑表现"], "unmatched_symptoms": []},
                {"name": "Social Communication Disorder", "confidence": "低",
                 "matching_symptoms": ["社交回避"], "unmatched_symptoms": ["特定场合不说话", "焦虑表现"]},
            ],
            "key_symptoms": ["特定场合不说话", "社交回避", "焦虑表现"],
            "analysis_summary": "患者在特定社交场合持续无法说话，伴有明显焦虑，高度提示选择性缄默症。"
        },
        "risk_assessment": {
            "risk_level": "中", "risk_score": 5,
            "dimensions": {
                "communication_impact": {"level": "中", "description": "在特定场合沟通完全受限"},
                "social_impact": {"level": "高", "description": "社交回避严重影响同伴关系"},
                "academic_impact": {"level": "中", "description": "课堂上不发言影响参与度"},
                "emotional_impact": {"level": "中", "description": "焦虑情绪持续存在"},
            },
            "risk_factors": ["社交广泛受限", "焦虑持续"],
            "protective_factors": ["家庭中语言正常", "年龄小可塑性强"],
            "urgency": "尽快干预",
            "recommendation": "建议启动渐进暴露和认知行为疗法，家校协作建立安全感。"
        },
        "rehab_plan": {
            "diagnosis_summary": "选择性缄默症",
            "rehab_plan": {
                "overall_goal": "逐步恢复社交场合的言语表达，降低焦虑水平",
                "duration_estimate": "3-6个月",
                "phases": [
                    {"phase": "初期建立安全感", "duration": "2-3周",
                     "objectives": ["建立信任关系", "非言语互动"],
                     "activities": [
                         {"name": "非言语沟通训练", "description": "从点头、手势开始", "frequency": "每日10分钟", "method": "在安全环境中用非言语方式回应"},
                         {"name": "渐进暴露", "description": "逐步引入陌生人", "frequency": "每周2次", "method": "从家人→熟悉老师→陌生人阶梯式推进"},
                     ]},
                    {"phase": "言语激活", "duration": "4-6周",
                     "objectives": ["在更多场合发声", "降低语言焦虑"],
                     "activities": [
                         {"name": "录音回听", "description": "录下自己说话并回听", "frequency": "每周3次", "method": "先录制独处时的声音，逐步在有小量听众时录音"},
                         {"name": "认知行为疗法", "description": "识别并挑战焦虑思维", "frequency": "每周1次", "method": "通过思维记录表和放松训练降低说话恐惧"},
                     ]},
                    {"phase": "社交泛化", "duration": "4-6周",
                     "objectives": ["在自然社交中说话", "维持对话"],
                     "activities": [
                         {"name": "社交技能训练", "description": "在小组中练习", "frequency": "每周2次", "method": "2-3人小组进行结构化对话练习"},
                         {"name": "家校协作", "description": "统一策略", "frequency": "持续", "method": "建立家校一致的沟通期待和奖励机制"},
                     ]},
                ]
            },
            "home_practice": ["家长创造低压力说话机会", "避免强迫说话", "积极强化任何发声尝试", "与学校保持沟通"],
            "follow_up": {"next_assessment": "4周后", "monitoring_indicators": ["发声场合数量", "焦虑自评分数", "社交互动频率"]},
            "precautions": ["绝不强迫或施压要求说话", "避免在他人面前强调不说话的行为", "关注共病焦虑的评估"]
        },
        "rag_sources": [{"id": "KB004", "disease": "Selective Mutism"}, {"id": "KB007", "disease": "Social Communication Disorder"}]
    }


def _stuttering_response(data):
    return {
        "patient": data,
        "symptom_analysis": {
            "possible_conditions": [
                {"name": "Stuttering (Fluency Disorder)", "confidence": "高",
                 "matching_symptoms": ["重复音节或词语", "延长发音", "说话紧张", "回避说话场景"], "unmatched_symptoms": []},
            ],
            "key_symptoms": ["重复音节或词语", "延长发音", "说话紧张", "回避说话场景"],
            "analysis_summary": "典型口吃表现，伴有说话紧张和场景回避，需及早干预防止慢性化。"
        },
        "risk_assessment": {
            "risk_level": "中", "risk_score": 5,
            "dimensions": {
                "communication_impact": {"level": "高", "description": "频繁口吃显著影响流畅沟通"},
                "social_impact": {"level": "中", "description": "回避说话场景影响社交参与"},
                "academic_impact": {"level": "中", "description": "课堂发言受影响"},
                "emotional_impact": {"level": "高", "description": "说话紧张和回避暗示情绪困扰"},
            },
            "risk_factors": ["家族口吃史", "伴有情绪困扰"],
            "protective_factors": ["有治疗意识", "家庭支持"],
            "urgency": "尽快干预",
            "recommendation": "建议启动流畅塑形和口吃修正训练，关注情绪脱敏。"
        },
        "rehab_plan": {
            "diagnosis_summary": "口吃（言语流畅性障碍），有家族史",
            "rehab_plan": {
                "overall_goal": "提高言语流畅度，降低说话恐惧",
                "duration_estimate": "6-12个月",
                "phases": [
                    {"phase": "流畅塑形训练", "duration": "4-6周",
                     "objectives": ["掌握流畅技术", "降低语速"],
                     "activities": [
                         {"name": "慢速说话练习", "description": "使用轻柔起音和连读", "frequency": "每日15分钟", "method": "从单句慢速朗读开始，逐步延长到段落"},
                         {"name": "呼吸训练", "description": "腹式呼吸和气息控制", "frequency": "每日2次", "method": "平躺感受腹式呼吸→坐位→站立→说话时应用"},
                     ]},
                    {"phase": "口吃修正训练", "duration": "4-8周",
                     "objectives": ["学会处理口吃时刻", "减少恐惧"],
                     "activities": [
                         {"name": "口吃修正技术", "description": "在口吃时以可控方式完成", "frequency": "每周2次", "method": "练习主动口吃降低恐惧"},
                         {"name": "脱敏训练", "description": "逐步面对说话恐惧", "frequency": "每周1次", "method": "从低压力→高压力场景层级暴露"},
                     ]},
                    {"phase": "维持与巩固", "duration": "8-12周",
                     "objectives": ["日常中维持流畅", "自信表达"],
                     "activities": [
                         {"name": "真实场景练习", "description": "在打电话、自我介绍等场景练习", "frequency": "每周3次", "method": "先预演后实战，记录成功经验"},
                     ]},
                ]
            },
            "home_practice": ["家长不做消极回应（不催促、不替说）", "创造轻松说话环境", "记录流畅说话的时刻", "规律运动降低整体焦虑"],
            "follow_up": {"next_assessment": "6周后", "monitoring_indicators": ["口吃频率", "说话自评焦虑", "回避场景数量"]},
            "precautions": ["家长不以焦虑态度回应口吃", "不要求慢慢说或想好再说", "注意共病社交焦虑"]
        },
        "rag_sources": [{"id": "KB008", "disease": "Stuttering"}, {"id": "KB002", "disease": "Speech Sound Disorder"}]
    }


if __name__ == "__main__":
    import uvicorn
    print("Demo server starting at http://localhost:8000")
    print("No LLM API key needed — using mock responses")
    uvicorn.run(app, host="0.0.0.0", port=8000)
