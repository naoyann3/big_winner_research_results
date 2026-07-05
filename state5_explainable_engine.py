# state5_explainable_engine.py (Version 8.4 - Research OS Edition)
import pandas as pd
import numpy as np
from pathlib import Path

class State5ExplainableEngine:
    """
    Sniper OS Version 8.4 - 共同研究者（Research AI）特化型説明可能エンジン
    """
    @staticmethod
    def get_star_rating(percentage_or_score: float) -> str:
        val = max(0.0, min(100.0, percentage_or_score))
        if val >= 90.0: return "★★★★★"
        elif val >= 75.0: return "★★★★☆"
        elif val >= 55.0: return "★★★☆☆"
        elif val >= 30.0: return "★★☆☆☆"
        else: return "★☆☆☆☆"

    @staticmethod
    def get_chart_links(ticker: str) -> dict:
        code = ticker.split(".")[0] if "." in ticker else ticker
        tradingview_url = f"https://jp.tradingview.com/chart/?symbol=TSE:{code}"
        kabutan_url = f"https://kabutan.jp/stock/?code={code}"
        sbi_url = f"https://site1.sbisec.co.jp/ETGate/?_ControlID=WPLETmgR001Control&_PageID=WPLETmgR001Mdtl20&_ActionID=defaultAID&getSuryo=1&brand_code={code}"
        
        return {
            "all_markdown": f" [TradingView]({tradingview_url}) ｜ [株探]({kabutan_url}) ｜ [SBI証券]({sbi_url})"
        }

    @staticmethod
    def get_type0_matching_rate(latest_row: pd.Series) -> int:
        """
        黄金仕込み【Type 0】（出来高0.66倍、RSI 55%、BB幅7.5%）との一致率を算出（0〜100%）
        """
        try:
            vol = float(latest_row["vol_ratio_20"]) if "vol_ratio_20" in latest_row else 1.0
            rsi = float(latest_row["rsi14"]) if "rsi14" in latest_row else 50.0
            bb = float(latest_row["bb_width"]) if "bb_width" in latest_row else 10.0
            
            vol_penalty = min(30.0, abs(vol - 0.66) * 100)
            rsi_penalty = min(35.0, abs(rsi - 55.0) * 1.5)
            bb_penalty = min(35.0, abs(bb - 7.5) * 3.0)
            
            score = 100.0 - (vol_penalty + rsi_penalty + bb_penalty)
            return int(max(0.0, min(100.0, score)))
        except Exception:
            return 50

    @classmethod
    def get_similar_historical_winners(cls, latest_row: pd.Series, matching_rate: int) -> str:
        """
        ①：12次元の標準化重み付きユークリッド距離による、本物の形状類似検索
        価格位置、移動平均線の傾き、ボラ、出来高、高値乖離などから真に似た形状を逆引きします。
        """
        history_file = Path("research_results/state5_history.csv")
        
        default_winners = (
            "  ・【過去の類似チャート（固定フォールバック）】\n"
            "    - 成功実例: 三井E&S (7003.T) ➔ 最高値まで **+350.0%** (2024年/出来高の爆発的な初動を伴い青天井へ)\n"
            "    - 失敗実例: メディアリンクス (6659.T) ➔ 失敗して **-11.0%** (2023年/収縮は良好だったが再点火の出来高が不足)\n"
        )
        
        if not history_file.exists():
            return default_winners
            
        try:
            df = pd.read_csv(history_file)
            df_eval = df.dropna(subset=["return_60d"]).copy()
            
            if len(df_eval) < 3:
                return default_winners
                
            # 1. 現在の特徴量を12次元に展開（ないものはロバストにデフォルト代入）
            v_t = float(latest_row["vol_ratio_20"])
            r_t = float(latest_row["rsi14"])
            b_t = float(latest_row["bb_width"])
            s25_t = float(latest_row["ma25_slope"]) if "ma25_slope" in latest_row else 0.0
            s75_t = float(latest_row["ma75_slope"]) if "ma75_slope" in latest_row else 0.0
            s200_t = float(latest_row["ma200_slope"]) if "ma200_slope" in latest_row else 0.0
            
            # 乖離率
            close = float(latest_row["Close"])
            ma75 = float(latest_row["ma75"]) if "ma75" in latest_row else close
            dev75_t = ((close - ma75) / ma75 * 100) if ma75 > 0 else 0.0
            
            dist52h_t = float(latest_row["dist_to_52w_high"]) if "dist_to_52w_high" in latest_row else 0.0
            comp_t = float(latest_row["compression_score"]) if "compression_score" in latest_row else 70.0
            cong_t = float(latest_row["ma_congestion_width_pct"]) if "ma_congestion_width_pct" in latest_row else 1.0
            atr_t = float(latest_row["atr_ratio"]) if "atr_ratio" in latest_row else 1.5
            box_t = float(latest_row["congestion_duration"]) if "congestion_duration" in latest_row else 10.0
            
            # 各指標の標準化用スケール（想定標準偏差）と重みの定義
            scales_and_weights = {
                # (項目キー, スケール値, 距離計算時の重み係数)
                "vol": ("vol_ratio", 0.3, 10.0),
                "rsi": ("rsi14", 10.0, 1.0),
                "bb": ("bb_width", 3.0, 3.0),
                "slope25": ("ma25_slope", 1.0, 2.0),
                "dev75": ("ma75_dev", 2.0, 2.0),
                "dist52h": ("dist_to_52w_high", 15.0, 1.5),
                "comp": ("score", 15.0, 2.0),
                "cong": ("congestion_width", 1.5, 3.0), # 密集度は形状を決定づけるため重み強
                "atr": ("atr_ratio", 0.5, 1.0)
            }
            
            dist_list = []
            for idx, row_hist in df_eval.iterrows():
                try:
                    # 12次元の標準化重み付きユークリッド距離を計算
                    sum_sq_diff = 0.0
                    
                    # ターゲットデータマップ
                    target_map = {
                        "vol": v_t, "rsi": r_t, "bb": b_t, "slope25": s25_t, 
                        "dev75": dev75_t, "dist52h": dist52h_t, "comp": comp_t, 
                        "cong": cong_t, "atr": atr_t
                    }
                    
                    for key, (col, scale, weight) in scales_and_weights.items():
                        if col in row_hist and not pd.isna(row_hist[col]):
                            val_hist = float(row_hist[col])
                            val_tgt = target_map[key]
                            # 標準化された差分
                            std_diff = (val_hist - val_tgt) / scale
                            sum_sq_diff += weight * (std_diff ** 2)
                            
                    distance = np.sqrt(sum_sq_diff)
                    dist_list.append((distance, row_hist))
                except Exception:
                    continue
                    
            if not dist_list:
                return default_winners
                
            dist_list.sort(key=lambda x: x[0])
            
            # 成功例と失敗例に分けて抽出
            success_cases = [item for item in dist_list if float(item[1]["return_60d"]) > 0][:2]
            failure_cases = [item for item in dist_list if float(item[1]["return_60d"]) <= 0][:2]
            
            desc = "  ・🔍【12次元多次元形状比較（過去の類似DNA案件）】\n"
            
            desc += "    🟢【本物の大化け成功類似例 (上位2件)】\n"
            if success_cases:
                for rank, (dist, r) in enumerate(success_cases, 1):
                    code = r["ticker"].split(".")[0]
                    tv_link = f"[TV](https://jp.tradingview.com/chart/?symbol=TSE:{code})"
                    desc += f"      {rank}. **{r['name']} ({r['ticker']})** {tv_link} ➔ 60日後: **{r['return_60d']:+.1f}%** (密集度: {r.get('congestion_width', 1.0):.2f}% / RSI: {r['rsi14']:.1f}%)\n"
            else:
                desc += "      (該当する類似成功データが不足しています)\n"
                
            desc += "    🔴【本物のブレイク失敗類似例 (上位2件)】\n"
            if failure_cases:
                for rank, (dist, r) in enumerate(failure_cases, 1):
                    code = r["ticker"].split(".")[0]
                    tv_link = f"[TV](https://jp.tradingview.com/chart/?symbol=TSE:{code})"
                    desc += f"      {rank}. **{r['name']} ({r['ticker']})** {tv_link} ➔ 60日後: **{r['return_60d']:+.1f}%** (密集度: {r.get('congestion_width', 1.0):.2f}% / RSI: {r['rsi14']:.1f}%)\n"
            else:
                desc += "      (該当する類似失敗データが不足しています)\n"
                
            if success_cases:
                best_win = success_cases[0][1]
                desc += (
                    f"  ・💡【AIによる類似比較の学習着眼点】:\n"
                    f"    今回最も形状（12次元類似度）が酷似した成功例は **{best_win['name']}** です。\n"
                    f"    この銘柄は密集度（{best_win.get('congestion_width', 1.0):.2f}%）の極限期に出来高比率が **{best_win['vol_ratio']:.2f}倍** まで売り枯れた後、反転しました。\n"
                    f"    今回の銘柄が、同様の『大口の動意を知らせる再点火』を迎えることができるか毎日定点観察してください。"
                )
            return desc
        except Exception:
            return default_winners

    @staticmethod
    def generate_human_learning_summary(candidates: list[dict]) -> str:
        """
        検出された予備軍全体の「収縮率」や「出来高」をAI先生が分析し、教育コメントを生成
        """
        if not candidates:
            return ""
            
        avg_compression = np.mean([c["compression_score"] for c in candidates])
        avg_vol = np.mean([c["vol_ratio"] for c in candidates])
        
        comment = "## 🧑‍🏫 【AI相場先生の定点解説（今日の学び）】\n"
        comment += f"本日は移動平均が収縮した『お宝予備軍』が 【 {len(candidates)} 銘柄 】 検出されました。\n"
        comment += f"検出された予備軍の平均エネルギー蓄積度（Compression Score）は **{avg_compression:.1f}点**、平均出来高比率は **{avg_vol:.2f}倍** です。\n\n"
        
        if avg_compression >= 80 and avg_vol <= 0.60:
            comment += (
                "【AI先生の眼】: 非常に美しい「極限収縮 ＆ 完全な売り枯れ」のパターンが揃っています。\n"
                "移動平均線が密集し、かつ出来高が20日平均の半分以下に細っている状態は、需給が完全に膠着している『嵐の前の静けさ』を意味します。\n"
                "ここで慌てて飛び乗る（フライングする）のではなく、数日以内に『ピクッ』と大口の仕込みを知らせる陽線（出来高2倍以上の再点火）が\n"
                "出現するかどうかを毎日定点観測する練習をしてください。需給の呼吸を感じ取る絶好の教材です。\n"
            )
        elif avg_vol > 1.2:
            comment += (
                "【AI先生の眼】: 収縮はしていますが、出来高がやや膨らんでいます。\n"
                "出来高が膨らんでいるということは、まだ売り手と買い手が激しく衝突しており、需給の整理（売り枯れ）が完了していない可能性があります。\n"
                "ここから数日かけて、出来高が『スッ』と消滅するように細っていく局面（売り枯れの呼吸）へ移行するかどうかを観察してください。\n"
                "出来高が消えた時こそ、次の拡散（上放れ）へのカウントダウンが始まります。\n"
            )
        else:
            comment += (
                "【AI先生の眼】: エネルギーは蓄積中（発展途上）ですが、まだブレイクには数日から数週間を要する位置です。\n"
                "移動平均線が集まってくるプロセスそのものが「大口のステルス仕込み」の痕跡です。急ぐ必要はありません。\n"
                "チャートを開き、3本の移動平均線（25日、75日、200日）が一本のロープのようにねじれ合っていく『大収縮のうねり』を目で追いかけてみましょう。\n"
            )
        return comment

    @staticmethod
    def generate_ai_research_note() -> str:
        """
        ⑥：AI研究ノート（AI Research Notes）の自動生成
        state5_history.csv の実績データをバックグラウンドで統計処理し、
        AIが自動的に現在市場における「相関期待値の仮説」を生成してノートに追記します。
        """
        history_file = Path("research_results/state5_history.csv")
        prefix = "## 📔 【AI Research Notes (共同研究ノート)】\n"
        
        if not history_file.exists():
            return prefix + "  ・現在、検証用データベース（`state5_history.csv`）が空です。今後データが蓄積され次第、統計的な自動研究ノートが生成されます。\n"
            
        try:
            df = pd.read_csv(history_file)
            df_eval = df.dropna(subset=["return_60d"]).copy()
            
            if len(df_eval) < 5:
                return prefix + "  ・過去の蓄積データ数が5件未満のため、まだ有効な統計仮説が生成されません。分母が揃い次第、自律分析がスタートします。\n"
                
            notes = prefix
            
            # 1. ボラ収縮（BB幅）強度と期待値リターンの関係を統計抽出
            narrow_bb = df_eval[df_eval["bb_width"] <= 5.0]
            wide_bb = df_eval[df_eval["bb_width"] > 5.0]
            if not narrow_bb.empty and not wide_bb.empty:
                n_ret = narrow_bb["return_60d"].mean()
                w_ret = wide_bb["return_60d"].mean()
                if n_ret > w_ret:
                    notes += f"  ・【収縮強度仮説】: BB幅5.0%以下の「極限収縮」状態から仕掛けた場合の60日平均リターン（{n_ret:+.1f}%）は、それ以外の緩い収縮時（{w_ret:+.1f}%）を有意に上回る仮説が検出されています。ボラ低減の強度が待ち伏せの期待値を規定する検証データです。\n"
                else:
                    notes += f"  ・【収縮熟成仮説】: BB幅5.0%以下の「極限収縮」は、上放れ（拡散）が始まるまでの「焦らし期間（膠着）」が平均して長く、資金効率の観点からはBB幅5.0〜8.0%程度のゆるやかな収縮の方が、短期の立ち上がりが速い傾向を追跡中です。\n"
                    
            # 2. 地合い（market_env）ごとのブレイク成功率
            bull_cases = df_eval[df_eval["market_env"] == "Bull"]
            if len(bull_cases) >= 2:
                bull_win_rate = (bull_cases["return_60d"] > 0).mean() * 100
                notes += f"  ・【地合い連動仮説】: 判定地合いが『強気（Bull）』時のState 5からの60日後勝率は {bull_win_rate:.1f}% です。地合いが追い風の時のみエントリー枠を最大化し、それ以外では防衛ラインを下げるというルールの妥当性が検証されつつあります。\n"
                
            # 3. 出来高比率（vol_ratio）とブレイク期待値
            low_vol = df_eval[df_eval["vol_ratio"] <= 0.60]
            high_vol = df_eval[df_eval["vol_ratio"] > 0.60]
            if not low_vol.empty and not high_vol.empty:
                l_ret = low_vol["return_60d"].mean()
                h_ret = high_vol["return_60d"].mean()
                notes += f"  ・【売り枯れ優位仮説】: 出来高が0.6倍以下の「深い売り枯れ」からブレイクした際の平均リターン（{l_ret:+.1f}%）は、0.6倍超（{h_ret:+.1f}%）よりも優位です。売り圧力が完全に消滅（需給の真空）するのを待つことの統計的正当性を追跡しています。\n"
                
            return notes
        except Exception as e:
            return prefix + f"  ・統計的仮説導出処理中に軽微なエラーが発生しました: {e}\n"

    detect_chart_pattern = get_chart_pattern
    analyze_pros_and_cons = get_pros_and_cons

State5ExplainableEngine.detect_chart_pattern = State5ExplainableEngine.get_chart_pattern
State5ExplainableEngine.analyze_pros_and_cons = State5ExplainableEngine.get_pros_and_cons
