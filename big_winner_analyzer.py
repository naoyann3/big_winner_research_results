[Step 1: EventFinder]      ──> データベースから「大化け株のDay 0」を特定
         │
[Step 2 & 3: FeatureExtractor] ──> Day 0の前後の株価を切り出し、
         │                         特徴量計算関数（Plugin）を順次適用して統合
         ▼
[Step 4: PatternAnalyzer]  ──> 統計、ヒートマップ、重要度分析（LightGBMなど）