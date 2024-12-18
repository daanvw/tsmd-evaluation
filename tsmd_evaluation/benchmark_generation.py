import pandas as pd
import numpy as np

def generate_tsmd_benchmark_ts(df, g=2):
    freqs   = df['label'].value_counts()
    
    classes = freqs.index
    classes = classes[freqs > 1]

    if len(classes) < g:
        ValueError("TODO")
    
    # Pick the repeating classes randomly
    repeating_classes = np.random.choice(classes, size=g, replace=False)  
    # Sample one instance of every non-repeating class. Then randomly order them
    X_non_repeating = df[~df['label'].isin(repeating_classes)].copy()
    X_non_repeating = X_non_repeating.groupby('label', group_keys=False).apply(lambda x: x.sample())
    X_non_repeating = X_non_repeating.sample(frac=1).reset_index(drop=True)
            
    # Sample at least two motifs for each repeating class
    X_repeating = df[df['label'].isin(repeating_classes)].reset_index(drop=True)
    motifs = X_repeating.groupby('label', group_keys=False).apply(lambda x: x.sample(n=2))
    
    # Then complete the GT motifs by randomly sampling other motifs from repeating classes
    other_motifs = X_repeating[~X_repeating.apply(tuple, 1).isin(motifs.apply(tuple, 1))]
    other_motifs = other_motifs.sample(n=min(len(other_motifs), max(0, len(X_non_repeating)+1-len(motifs))), replace=False)

    all_motifs = pd.concat((motifs, other_motifs))    
    all_motifs = all_motifs.sample(frac=1).reset_index(drop=True)
            
    gt = {c: [] for c in repeating_classes}
    ts = []

    # Concatenate instances, alternating between a non-repeating and a repeating class, until no instances left
    curr   = 0
    for i in range(min(len(X_non_repeating), len(all_motifs) - 1)):
        # Motif
        motif, label, l = all_motifs.iloc[i]
        ts.append(motif)
        gt[label].append((curr, curr+l))
        curr += l

        # Non-motif
        instance, _, l = X_non_repeating.iloc[i]
        ts.append(instance)
        curr += l
    
    motif, label, l = all_motifs.iloc[-1]
    ts.append(motif)
    gt[label].append((curr, curr+l))    
    return np.vstack(ts), gt

def convert_X_y_to_df(X, y):
    time_series = [x.T for x in X]
    lengths = [len(ts) for ts in time_series]
    df = pd.DataFrame({"ts": time_series, "label": y, "length": lengths})
    return df

def generate_tsmd_benchmark_dataset(df, N, g_min, g_max):    
    # Generate time series
    benchmark_ts = []
    gts = []
    for _ in range(N):
        
        # Sample a number of motif sets
        g = np.random.randint(g_min, g_max+1)
    
        ts, gt = generate_tsmd_benchmark_ts(df, g=g)
        benchmark_ts.append(ts)
        gts.append(gt)
        
    benchmark_dataset = pd.DataFrame({'ts': benchmark_ts, 'gt': gts})
    return benchmark_dataset