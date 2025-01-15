import numpy as np
import random
import multiprocessing
import multiprocessing
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

##################################################
######### introduce errors in the reads ##########
##################################################       

def introduce_errors_with_labels(sequence, label, mismatch_rate, 
                                 insertion_rate, deletion_rate, 
                                 max_insertions=5):
    error_sequence = []
    error_labels = []
    
    for base, lbl in zip(sequence, label):
        insertion_count = 0
        r = np.random.random()
        
        if r < mismatch_rate:
            error_sequence.append(np.random.choice([b for b in 'ATCG' if b != base]))
            error_labels.append(lbl)
            
        elif (r < mismatch_rate + insertion_rate):
            rerun = "yes"
            while rerun == "yes":
                error_sequence.append(np.random.choice(list('ATCG')))
                error_labels.append(lbl)
                
                insertion_count += 1
                
                r = np.random.random()

                if (r < mismatch_rate + insertion_rate) and (insertion_count <= max_insertions):
                    rerun = "yes"
                else:
                    rerun = "no"
                    
            error_sequence.append(base)
            error_labels.append(lbl)
            
        elif r < mismatch_rate + insertion_rate + deletion_rate:
            continue
        else:
            error_sequence.append(base)
            error_labels.append(lbl)
            
    return ''.join(error_sequence), error_labels

##################################################
###### Simulate single cDNA element reads ########
##################################################

def simulate_complete_reads(seq_order, seqs, num_reads=15000, max_polya_length=28, 
                            min_cdna_length=500, max_cdna_length=600, mismatch_rate=0.044, 
                            insertion_rate=0.046116500187765, deletion_rate=0.0612981959469103):
    reads = []
    labels = []
    
    for _ in range(num_reads):
        
        read_segments = []
        label_segments = []

        random_length_1 = np.random.randint(0, 40)
        random_1 = ''.join(np.random.choice(list('ATCG')) for _ in range(random_length_1))

        read_segments.append(random_1)
        label_segments.append(['random'] * random_length_1)

        for i in range(len(seq_order)):
            if (seqs[i][0] != "N"):
                read_segments.append(seqs[i])
                label_segments.append([seq_order[i]] * len(seqs[i]))

            elif (seqs[i][0] == "N"):
                if (seqs[i][1] != "N"):
                    read_segments.append(''.join(np.random.choice(list('ATCG')) for _ in range(int(seqs[i][1]))))
                    label_segments.append([seq_order[i]] * int(seqs[i][1]))
                else:
                    cDNA_length = np.random.randint(min_cdna_length, max_cdna_length+1)
                    cDNA = ''.join(np.random.choice(list('ATCG')) for _ in range(cDNA_length))
                    read_segments.append(cDNA)
                    label_segments.append([seq_order[i]] * cDNA_length)

            elif (seqs[i][0] == "T"):
                polyA = 'T' * np.random.randint(10, max_polya_length+1)
                read_segments.append(polyA)
                label_segments.append([seq_order[i]] * len(polyA))

            elif (seqs[i][0] == "A"):
                polyA = 'A' * np.random.randint(10, max_polya_length+1)
                read_segments.append(polyA)
                label_segments.append([seq_order[i]] * len(polyA))
        
        random_length_2 = np.random.randint(0, 40)
        random_2 = ''.join(np.random.choice(list('ATCG')) for _ in range(random_length_2))

        read_segments.append(random_2)
        label_segments.append(['random'] * random_length_2)
        
        error_read_segments = []
        error_label_segments = []
        
        for seg, lbl in zip(read_segments, label_segments):
            err_seg, err_lbl = introduce_errors_with_labels(seg, lbl, mismatch_rate, insertion_rate, deletion_rate)
            error_read_segments.append(err_seg)
            error_label_segments.extend(err_lbl)

        reads.append(''.join(read_segments))
        labels.append([item for sublist in label_segments for item in sublist])

        reads.append(''.join(error_read_segments))
        labels.append(error_label_segments)
    
    return reads, labels

##################################################
###### parallelized simulation of reads ##########
##################################################  

# Define a wrapper function to set the random seed before calling the actual function
def wrapper_func(func, *args, **kwargs):
    seed = args[0]
    random.seed(int(seed))  # Ensure seed is an integer
    np.random.seed(int(seed))
    return func(*args[1:], **kwargs)

def parallel_generate_reads(seq_order, seqs, num_reads, portion, max_polya_length, 
                            min_cdna_length, max_cdna_length, mismatch_rate, 
                            insertion_rate, deletion_rate, num_processes):
    
    pool = multiprocessing.Pool(processes=num_processes)
    chunk_size = num_reads // num_processes

    # Generate a list of unique seeds for each process
    seeds = [random.randint(0, 999999) for _ in range(num_processes)]
    
    if portion == "complete":
        results = pool.starmap(wrapper_func, [(simulate_complete_reads, seed, chunk_size, max_polya_length, 
                                               min_cdna_length, max_cdna_length, mismatch_rate, insertion_rate, 
                                               deletion_rate) 
                                for seed in seeds])  
    pool.close()
    pool.join()

    reads = []
    labels = []

    for result in results:
        reads.extend(result[0])
        labels.extend(result[1])

    return reads, labels

def simulate_training_reads(seq_order, seqs, type, threads, mismatch_rate, insertion_rate, deletion_rate):
    if type != "array":
        complete_reads, complete_labels = parallel_generate_reads(seq_order=seq_order, 
                                                                  seqs=seqs,
                                                                  num_reads=15000, 
                                                                  portion="complete",
                                                                  max_polya_length=28,
                                                                  min_cdna_length=500,
                                                                  max_cdna_length=600,
                                                                  mismatch_rate = mismatch_rate, 
                                                                  insertion_rate = insertion_rate, 
                                                                  deletion_rate = deletion_rate,
                                                                  num_processes=threads) 
    return  complete_reads, complete_labels

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join(complement[base] for base in reversed(seq))

nucleotide_to_id = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}

def encode_sequence(read, nucleotide_to_id):
    return [nucleotide_to_id[n] for n in read]

def prepare_training_data(seq_order, seqs, type, threads, mismatch_rate, insertion_rate, deletion_rate):

    training_reads, training_labels = simulate_training_reads(seq_order, seqs, type, threads, mismatch_rate, insertion_rate, deletion_rate)
    
    ## add reverse complements
    reverse_complement_reads = [reverse_complement(read) for read in training_reads]
    reversed_labels = [list(reversed(label)) for label in training_labels]
    
    all_reads = reverse_complement_reads + training_labels 
    all_labels = reversed_labels + training_labels

    X = [encode_sequence(read, nucleotide_to_id) for read in all_reads]
    
    # Flatten all labels into a single list to fit LabelBinarizer
    Y = [label for sublist in all_labels for label in sublist]
    
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(Y)
    
    Y_encoded = [label_binarizer.transform(seq) for seq in all_labels]
    # Determine the max sequence length
    max_seq_len = max([len(seq) for seq in X])
    
    # Padding sequences to have the same length
    X_padded = pad_sequences(X, padding='post', dtype='float32', 
                             maxlen=max_seq_len, value=0)
    Y_padded = pad_sequences(Y_encoded, padding='post', dtype='float32', 
                             maxlen=max_seq_len, value=0)

    X_train, X_val, Y_train, Y_val = train_test_split(X_padded, Y_padded, test_size=0.2)

    return X_train, X_val, Y_train, Y_val, label_binarizer, max_seq_len

