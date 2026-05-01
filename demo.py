"""
Demo mode — runs the full pipeline WITHOUT an LLM API key.

Uses mock agent responses to demonstrate the system flow end-to-end.
Great for understanding the architecture before connecting a real LLM.
"""

import json
from rag.retriever import RAGRetriever
from utils.report import to_structured_json


class MockLLMAgent:
    """Simulates an LLM agent with deterministic mock responses."""

    def __init__(self, name: str):
        self.name = name


class MockSymptomAnalyzer(MockLLMAgent):
    def run(self, patient_data, rag_context):
        return {
            "possible_conditions": [
                {
                    "name": "Developmental Language Disorder (DLD)",
                    "confidence": "高",
                    "matching_symptoms": ["语言表达困难", "发音不清"],
                    "unmatched_symptoms": ["注意力不集中"],
                },
                {
                    "name": "ADHD",
                    "confidence": "中",
                    "matching_symptoms": ["注意力不集中"],
                    "unmatched_symptoms": ["发音不清", "语言表达困难"],
                },
            ],
            "key_symptoms": ["发音不清", "语言表达困难", "注意力不集中"],
            "analysis_summary": "患者表现为语言表达和语音方面的障碍，同时伴有注意力问题。"
            "主要考虑发育性语言障碍（DLD），需注意是否合并注意缺陷多动障碍（ADHD）。",
        }


class MockRiskAssessor(MockLLMAgent):
    def run(self, symptom_analysis, patient_data):
        return {
            "risk_level": "中",
            "risk_score": 5,
            "dimensions": {
                "communication_impact": {"level": "高", "description": "语言表达和发音均受影响，沟通能力明显受限"},
                "social_impact": {"level": "中", "description": "沟通困难可能导致社交回避"},
                "academic_impact": {"level": "中", "description": "注意力不集中影响课堂学习"},
                "emotional_impact": {"level": "低", "description": "暂无明显情绪问题表现"},
            },
            "risk_factors": ["语言功能的广泛性影响", "注意力问题可能加重语言障碍"],
            "protective_factors": ["年龄小，神经可塑性强", "无重大疾病史"],
            "urgency": "尽快干预",
            "recommendation": "建议尽快启动语言康复训练，同时评估注意力情况，必要时联合干预。",
        }


class MockRehabPlanner(MockLLMAgent):
    def run(self, patient_data, symptom_analysis, risk_assessment, rag_context):
        return {
            "diagnosis_summary": "发育性语言障碍（DLD），可能合并注意缺陷特征（需进一步评估）",
            "rehab_plan": {
                "overall_goal": "改善语言表达能力和语音清晰度，提升注意力持续水平",
                "duration_estimate": "3-6个月",
                "phases": [
                    {
                        "phase": "初期评估与基础训练",
                        "duration": "2-4周",
                        "objectives": ["完成详细语言评估", "建立训练关系", "基础口腔运动训练"],
                        "activities": [
                            {
                                "name": "口腔运动训练",
                                "description": "增强口腔肌肉协调性",
                                "frequency": "每日10分钟",
                                "method": "吹气球、舌头体操、唇部按摩",
                            },
                            {
                                "name": "听觉辨别训练",
                                "description": "训练区分正确与错误发音",
                                "frequency": "每周3次，每次15分钟",
                                "method": "最小对立体辨别游戏，如'ba/pa'配对",
                            },
                        ],
                    },
                    {
                        "phase": "核心语言训练",
                        "duration": "6-8周",
                        "objectives": ["扩展词汇量", "改善语音清晰度", "提升语句长度"],
                        "activities": [
                            {
                                "name": "词汇扩展练习",
                                "description": "从名词到动词到形容词逐步推进",
                                "frequency": "每日20分钟",
                                "method": "使用图卡命名、情景描述、词汇接龙游戏",
                            },
                            {
                                "name": "构音训练",
                                "description": "针对错误音素系统矫正",
                                "frequency": "每周2次，每次30分钟",
                                "method": "单音→音节→词语→句子渐进训练，使用多感官提示",
                            },
                            {
                                "name": "正念注意力训练",
                                "description": "提升注意力和自我调节",
                                "frequency": "每日5-10分钟",
                                "method": "专注呼吸练习、听觉注意游戏（如'听到拍手声就举手'）",
                            },
                        ],
                    },
                    {
                        "phase": "巩固与泛化",
                        "duration": "4-8周",
                        "objectives": ["在自然情境中运用语言", "维持注意力", "社交沟通提升"],
                        "activities": [
                            {
                                "name": "叙事训练",
                                "description": "提升篇章组织和叙事能力",
                                "frequency": "每周3次，每次20分钟",
                                "method": "使用图片故事卡引导讲述，逐步增加复杂度",
                            },
                            {
                                "name": "社交对话练习",
                                "description": "在游戏情境中练习对话轮流和话题维持",
                                "frequency": "每周2次",
                                "method": "角色扮演、合作桌游、话题讨论",
                            },
                        ],
                    },
                ],
            },
            "home_practice": [
                "家长每日与儿童进行15分钟慢速清晰对话，使用扩展性回应",
                "亲子共读时鼓励儿童描述图片内容",
                "利用日常场景（超市、公园）进行词汇和句子练习",
                "定时器辅助完成注意力任务（如5分钟拼图）",
            ],
            "follow_up": {
                "next_assessment": "4-6周后进行中期评估",
                "monitoring_indicators": [
                    "语音清晰度变化",
                    "平均语句长度（MLU）",
                    "注意力持续时间",
                    "社交参与度",
                ],
            },
            "precautions": [
                "训练需在儿童精力充沛时进行，避免疲劳状态下强迫训练",
                "注意ADHD共病的可能，若注意力问题持续应转诊评估",
                "家长参与需保持积极鼓励态度，避免批评纠正过多",
                "保持训练的游戏性和趣味性，维持儿童动机",
            ],
        }


class MockDispatcher:
    """Mock dispatcher that uses mock agents instead of real LLM calls."""

    def __init__(self, rag_retriever):
        self.rag = rag_retriever
        self.symptom_agent = MockSymptomAnalyzer("SymptomAnalysis")
        self.risk_agent = MockRiskAssessor("RiskAssessment")
        self.rehab_agent = MockRehabPlanner("RehabPlan")

    def run(self, patient_data):
        # RAG retrieval still works with real embeddings (or local fallback)
        query = f"{patient_data.get('symptoms', '')} 儿童 年龄{patient_data.get('age', '')}"
        rag_results = self.rag.retrieve(query)
        rag_context = self.rag.format_context(rag_results)

        print(f"  [RAG] Retrieved {len(rag_results)} knowledge entries")
        for r in rag_results:
            print(f"       → {r.get('id')}: {r.get('disease')}")

        symptom_analysis = self.symptom_agent.run(patient_data, rag_context)
        print(f"  [SymptomAnalysis] Found {len(symptom_analysis['possible_conditions'])} possible conditions")

        risk_assessment = self.risk_agent.run(symptom_analysis, patient_data)
        print(f"  [RiskAssessment] Risk level: {risk_assessment['risk_level']} (score: {risk_assessment['risk_score']})")

        rehab_plan = self.rehab_agent.run(patient_data, symptom_analysis, risk_assessment, rag_context)
        print(f"  [RehabPlan] Generated {len(rehab_plan['rehab_plan']['phases'])} training phases")

        return {
            "patient": patient_data,
            "symptom_analysis": symptom_analysis,
            "risk_assessment": risk_assessment,
            "rehab_plan": rehab_plan,
            "rag_sources": [{"id": r.get("id"), "disease": r.get("disease")} for r in rag_results],
        }


def main():
    print("=" * 60)
    print("  Medical Rehab Agent System — DEMO MODE")
    print("  (No LLM API key required)")
    print("=" * 60)

    patient_data = {
        "age": 10,
        "symptoms": "发音不清，语言表达困难，注意力不集中",
        "history": "无重大疾病史",
    }

    print(f"\n[Input] Patient: {json.dumps(patient_data, ensure_ascii=False)}")

    # Initialize RAG (uses local embedding fallback)
    print("\n[RAG] Initializing knowledge base (local embedding mode)...")
    retriever = RAGRetriever()
    retriever.initialize()

    # Run pipeline with mock agents
    print("\n[Pipeline] Running: RAG → SymptomAnalysis → RiskAssessment → RehabPlan...\n")
    dispatcher = MockDispatcher(retriever)
    result = dispatcher.run(patient_data)

    # Output
    structured_output = to_structured_json(result)
    print("\n" + "=" * 60)
    print("  Structured Output")
    print("=" * 60)
    print(structured_output)

    print("\n" + "=" * 60)
    print("  Full Agent Output (raw)")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
