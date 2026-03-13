def edit_distance(seq_a:list,seq_b:list)->int:
    n = len(seq_a)
    m = len(seq_b)

    dp = [[0]*(m+1) for _ in range (n+1)]

    for j in range(m+1):
        dp[0][j] = j

    for i in range(n+1):
        dp[i][0] = i 

    for i in range(1,n+1):
        for j in range(1,m+1):
            if seq_a[i-1] == seq_b[j-1]:
                dp[i][j] = dp[i-1][j-1]

            else:
                dp[i][j] = 1+ min(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1]
                )    

    return dp[n][m]

def cer(prediction:str,reference:str)->float:
    if len(reference) == 0:
        return 0.0 if len(prediction) ==0 else 1.0
    pred_chars = list(prediction)
    ref_chars = list(reference)

    dist = edit_distance(pred_chars,ref_chars)
    return dist / len(ref_chars)

def wer(prediction:str,reference:str)->float:
    ref_words = reference.strip().split()
    if len(ref_words) == 0:
        pred_words = prediction.strip().split()
        return 0.0 if len(pred_words) == 0 else 1.0

    pred_words = prediction.strip().split()

    dist = edit_distance(pred_words, ref_words)
    return dist / len(ref_words)

def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    assert len(predictions) == len(references), (
        f"Length mismatch: {len(predictions)} predictions , {len(references)} references"
    )

    total_cer = 0.0
    total_wer = 0.0
    perfect_cer = 0
    perfect_wer = 0
    per_sample = []

    for pred, ref in zip(predictions, references):
        sample_cer = cer(pred, ref)
        sample_wer = wer(pred, ref)

        total_cer += sample_cer
        total_wer += sample_wer

        if sample_cer == 0.0:
            perfect_cer += 1
        if sample_wer == 0.0:
            perfect_wer += 1

        per_sample.append({"cer": sample_cer, "wer": sample_wer})

    n = len(predictions)
    return {
        "cer": total_cer / max(1, n),
        "wer": total_wer / max(1, n),
        "num_samples": n,
        "num_perfect_cer": perfect_cer,
        "num_perfect_wer": perfect_wer,
        "perfect_cer_pct": perfect_cer / max(1, n) * 100,
        "perfect_wer_pct": perfect_wer / max(1, n) * 100,
        "per_sample": per_sample,
    }

def print_evaluation_report(metrics: dict, show_samples: int = 0, predictions: list = None, references: list = None):
    print("   Evaluation Report")
    print(f"  Samples evaluated:     {metrics['num_samples']}")
    print(f"  Character Error Rate (CER):  {metrics['cer']:.4f}  ({metrics['cer']*100:.2f}%)")
    print(f"  Word Error Rate (WER):       {metrics['wer']:.4f}  ({metrics['wer']*100:.2f}%)")
    print(f"  Perfect CER (0% error):  {metrics['num_perfect_cer']:4d} / {metrics['num_samples']}  ({metrics['perfect_cer_pct']:.1f}%)")
    print(f"  Perfect WER (0% error):  {metrics['num_perfect_wer']:4d} / {metrics['num_samples']}  ({metrics['perfect_wer_pct']:.1f}%)")

    if show_samples > 0 and predictions and references:
        print(f"\n  Sample Predictions (first {show_samples}):")
        for i in range(min(show_samples, len(predictions))):
            sample_metrics = metrics["per_sample"][i]
            print(f"  [{i+1}] CER: {sample_metrics['cer']*100:.1f}%  WER: {sample_metrics['wer']*100:.1f}%")
            print(f"    REF: {references[i]}")
            print(f"    PRD: {predictions[i]}")
