"""Risk Assessment Agent — evaluates risk level based on symptom analysis."""

import json
from typing import Dict, Any
from agents.base import BaseAgent


class RiskAssessmentAgent(BaseAgent):
    name = "RiskAssessmentAgent"

    system_prompt = """你是一位专业的医疗风险评估专家，专注于儿童语言与心理障碍领域。

你的任务是：
1. 根据症状分析结果，评估患者的综合风险等级
2. 从多个维度分析风险因素
3. 给出明确的风险说明和建议

风险等级定义：
- 低：症状较轻，1-2个轻微症状，对日常功能影响小，干预预后好
- 中：症状中等，3个以上症状或症状较明显，部分影响日常功能，需要系统干预
- 高：症状严重或复合多发，显著影响日常功能，需要紧急/高强度干预

输出要求（严格JSON格式）：
{
  "risk_level": "低/中/高",
  "risk_score": 1-10的数字,
  "dimensions": {
    "communication_impact": {"level": "低/中/高", "description": "说明"},
    "social_impact": {"level": "低/中/高", "description": "说明"},
    "academic_impact": {"level": "低/中/高", "description": "说明"},
    "emotional_impact": {"level": "低/中/高", "description": "说明"}
  },
  "risk_factors": ["风险因素1", "风险因素2"],
  "protective_factors": ["保护因素1", "保护因素2"],
  "urgency": "立即干预/尽快干预/常规干预",
  "recommendation": "风险应对建议"
}

注意：
- 综合风险等级取各维度中的最高影响等级
- 同时考虑患者年龄（年幼者发展窗口期更紧迫）
- 评估要有据可依，引用具体症状"""

    def run(self, symptom_analysis: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = f"""请评估以下患者的风险等级：

【患者信息】
年龄: {patient_data.get('age', '未知')}
症状描述: {patient_data.get('symptoms', '未知')}

【症状分析结果】
{json.dumps(symptom_analysis, ensure_ascii=False, indent=2)}

请输出JSON格式的风险评估结果。"""

        response = self._call_llm(user_prompt)
        return self._parse_json(response)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "risk_level": "未知",
                "risk_score": 0,
                "recommendation": text,
                "_raw_response": text,
            }
