"""Rehab Plan Agent — generates personalized rehabilitation plan."""

import json
from typing import Dict, Any
from agents.base import BaseAgent


class RehabPlanAgent(BaseAgent):
    name = "RehabPlanAgent"

    system_prompt = """你是一位资深儿童康复方案制定专家，擅长语言治疗和心理康复。

你的任务是：
1. 根据症状分析和风险评估结果，制定个性化康复训练方案
2. 方案需具体、可执行、分阶段
3. 充分利用知识库中的康复方法，结合患者情况调整

输出要求（严格JSON格式）：
{
  "diagnosis_summary": "诊断总结（注意：仅作参考，非正式诊断）",
  "rehab_plan": {
    "overall_goal": "总体康复目标",
    "duration_estimate": "预计周期",
    "phases": [
      {
        "phase": "阶段名称（如：初期评估与基础训练）",
        "duration": "持续时间（如：2-4周）",
        "objectives": ["目标1", "目标2"],
        "activities": [
          {
            "name": "训练活动名称",
            "description": "详细描述",
            "frequency": "频率（如：每日15分钟）",
            "method": "具体方法/步骤"
          }
        ]
      }
    ]
  },
  "home_practice": ["家庭练习建议1", "家庭练习建议2"],
  "follow_up": {
    "next_assessment": "下次评估时间建议",
    "monitoring_indicators": ["观察指标1", "观察指标2"]
  },
  "precautions": ["注意事项1", "注意事项2"]
}

注意：
- 方案要因人而异，不能简单复制知识库内容
- 考虑患者年龄选择适合的训练形式（儿童需要游戏化）
- 训练强度要与风险等级匹配
- 包含家庭练习建议（家长参与对儿童康复至关重要）"""

    def run(
        self,
        patient_data: Dict[str, Any],
        symptom_analysis: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        rag_context: str,
    ) -> Dict[str, Any]:
        user_prompt = f"""请为以下患者制定个性化康复方案：

【患者信息】
年龄: {patient_data.get('age', '未知')}
症状描述: {patient_data.get('symptoms', '未知')}
病史: {patient_data.get('history', '未知')}

【症状分析】
{json.dumps(symptom_analysis, ensure_ascii=False, indent=2)}

【风险评估】
{json.dumps(risk_assessment, ensure_ascii=False, indent=2)}

【知识库参考】
{rag_context}

请输出JSON格式的康复方案。"""

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
                "diagnosis_summary": "解析失败",
                "rehab_plan": {"overall_goal": "", "phases": []},
                "_raw_response": text,
            }
