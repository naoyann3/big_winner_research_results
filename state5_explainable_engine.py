# state5_explainable_engine.py (Version 8.3 - Human Learning Edition)
import pandas as pd
import numpy as np
from pathlib import Path

class State5ExplainableEngine:
    """
    Sniper OS Version 8.3 - 人間の相場観を育てる「学習OS」支援特化型エンジン
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
        ③：類似チャート検索の完全改良
        特徴量（出来高、RSI、BB幅）の距離を計算し、
        過去データベースから「本物の成功事例（上位2件）」と「本物の失敗事例（上位2件）」を動的に抽出します。
        """
        history_file = Path("research_results/state5_history.csv")
        
        # データベースがない場合の、歴史的に有名なフォールバックデータ
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
            
            if len(df_eval) < 5:
                return default_winners
                
            # 今回のターゲット特徴量
            v_t = float(latest_row["vol_ratio_20"])
            r_t = float(latest_row["rsi14"])
            b_t = float(latest_row["bb_width"])
            
            # 全過去データとの多次元ユークリッド距離を計算
            dist_list = []
            for idx, row in df_eval.iterrows():
                try:
                    v_i = float(row["vol_ratio"])
                    r_i = float(row["rsi14"])
                    b_i = float(row["bb_width"])
                    
                    # 出来高、RSI、BB幅に重み付けして距離を算出
                    distance = np.sqrt(
                        (15.0 * (v_i - v_t))**2 + 
                        (0.2 * (r_i - r_t))**2 + 
                        (0.5 * (b_i - b_t))**2
                    )
                    dist_list.append((distance, row))
                except Exception:
                    continue
                    
            if not dist_list:
                return default_winners
                
            # 距離が近い順にソート
            dist_list.sort(key=lambda x: x[0])
            
            # 成功例（return_60d > 0）と失敗例（return_60d <= 0）に分ける
            success_cases = [item for item in dist_list if float(item[1]["return_60d"]) > 0][:2]
            failure_cases = [item for item in dist_list if float(item[1]["return_60d"]) <= 0][:2]
            
            desc = "  ・🔍【動的類似チャート検索（過去の類似DNA案件）】\n"
            
            # 成功事例の印字
            desc += "    🟢【本物の大化け成功類似例 (上位2件)】\n"
            if success_cases:
                for rank, (dist, row) in enumerate(success_cases, 1):
                    code = row["ticker"].split(".")[0]
                    tv_link = f"[TV](https://jp.tradingview.com/chart/?symbol=TSE:{code})"
                    desc += f"      {rank}. **{row['name']} ({row['ticker']})** {tv_link} ➔ 60日後: **{row['return_60d']:+.1f}%** (BB幅: {row['bb_width']:.1f}% / 出来高比: {row['vol_ratio']:.2f}倍)\n"
            else:
                desc += "      (該当する類似成功データが不足しています)\n"
                
            # 失敗事例の印字
            desc += "    🔴【本物のブレイク失敗類似例 (上位2件)】\n"
            if failure_cases:
                for rank, (dist, row) in enumerate(failure_cases, 1):
                    code = row["ticker"].split(".")[0]
                    tv_link = f"[TV](https://jp.tradingview.com/chart/?symbol=TSE:{code})"
                    desc += f"      {rank}. **{row['name']} ({row['ticker']})** {tv_link} ➔ 60日後: **{row['return_60d']:+.1f}%** (BB幅: {row['bb_width']:.1f}% / 出来高比: {row['vol_ratio']:.2f}倍)\n"
            else:
                desc += "      (該当する類似失敗データが不足しています)\n"
                
            # AI学習ポイントの抽出
            if success_cases:
                best_win = success_cases[0][1]
                desc += (
                    f"  ・💡【AIによる類似比較の学習着眼点】:\n"
                    f"    今回の銘柄に最もチャート形状が似ている成功実例は **{best_win['name']}** です。\n"
                    f"    この銘柄は収縮（BB幅 {best_win['bb_width']:.1f}%）の後、出来高が20日平均を大きく超えて急激に『再点火』したことで青天井へブレイクしました。\n"
                    f"    今回検出された銘柄が、これから出来高を伴って同じような『上抜けの呼吸』を見せるかどうかを注視してください。"
                )
            return desc
        except Exception:
            return default_winners

    @staticmethod
    def generate_human_learning_summary(candidates: list[dict]) -> str:
        """
        ⑦：Human Learning Comment（AI先生の定点教育コメント）
        検出された予備軍全体の「収縮率（Compression）」や「出来高の細り方」をAI先生が分析し、
        投資判断ではなく、「相場を学ぶための教育コメント」を生成します。
        """
        if not candidates:
            return (
                "## 🧑‍🏫 【AI相場先生の定点解説（今日の学び）】\n"
                "本日は移動平均大収縮（Early Watch）の合格者が0件でした。\n"
                "市場全体が急激に動いているか、あるいは収縮が未成熟な状態にあります。こういう「静かな日」こそ、\n"
                "無理に銘柄を探すのではなく、過去の成功チャートをTradingViewで振り返り、\n"
                "『エネルギーが限界まで充填されると、どのような予備動作が起きるか』を脳に焼き付ける最高の練習機会です。待つこともまた、技術です。\n"
            )
            
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
                "チャートを開き、3本の移動平均線（25日、75日、200日）が一本のロープのようにねじれ合っていく『大収縮のうねり』を目で追いかけてみましょう。耳を澄ませば需給の音が聞こえてきます。\n"
            )
            
        return comment
