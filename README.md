# InsightSpike-AI — geDIG: 統一ゲージで洞察を測る

[![CI (Lite)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml)
[![CI (Unit)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-unit.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-unit.yml)

世界初の「Aha!モーメント」検出AI。知識グラフ上の構造コストと情報利得を単一ゲージで評価し、洞察（スパイク）を検出・制御する。

F = ΔEPC_norm − λ·ΔIG  （ΔIG = ΔH_norm + γ·ΔSP_rel）

## ⚡ 30秒で試す

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .

# その1: モックLLMで即動作（外部API不要）
python examples/public_quick_start.py

# その2: geDIG最小デモ（F, ΔEPC_norm, ΔIG を出力）
python examples/hello_insight.py
```

出力例（概略）:
```
F = -0.42  (ΔEPC_norm=0.15,  ΔIG=0.57,  spike=True)
```

## 🎯 2つの価値提案

- Phase 1（実装済み・今日から使える）
  - クエリ中心の局所サブグラフで ΔEPC/ΔIG を評価し、受容/保留/棄却・探索・バックトラックをイベント駆動で制御
  - 迷路: ステップ削減、RAG: PSZ準拠の精度・効率改善を目指す構成

- Phase 2（設計済み・共同研究を募集）
  - FEP–MDL ブリッジの枠組みにより、オフラインの大域再配線（GED_minを正則化/制約として活用）へ拡張
  - 数学的検証と大規模実験（10k+ノード）に向けた道筋を提示

## 🧭 ドキュメント

- QUICKSTART.md — 5分で始める（環境構築・最短実行）
- CONCEPTS.md — 用語と理論（ΔEPC/ΔIG, One‑Gauge, AG/DG, フェーズ）
- EXPERIMENTS.md — 迷路/RAG の再現入口（RAGは順次短縮化）
- 論文 v3（EPC基準）: docs/paper/geDIG_onegauge_improved_v3.tex
- 図（概念・結果）: docs/paper/figures/

## 🧪 最小API例（Public API）

```python
from insightspike.public import create_agent

agent = create_agent()  # 軽量モード既定
res = agent.process_question("geDIGを一文で？")
print(getattr(res, 'response', res.get('response', 'No response')))
```

## 実装のポイント（Phase 1）

- ΔEPC_norm: 「最小距離」ではなく、実際に適用した編集操作の正規化コスト（edit‑path cost; operational）
- ΔIG: ΔH_norm + γ·ΔSP_rel（SPは符号付き、固定ペア評価に対応）
- One‑Gauge制御: F が十分に小さいと“洞察スパイク”、二段ゲート（AG/DG）で判定を堅牢化

## ライセンス / コンタクト

- License: Apache-2.0
- 連絡先: miyauchikazuyoshi@gmail.com （コラボ歓迎：数理/実装/検証）
