import multiprocessing as mp
import numpy as np
# import polars as pl

################ collapse labels into order ################

def collapse_labels(arr, read_length):
    read = arr[0:read_length]
    collapsed_array = []
    count_dict = {}
    indices_dict = {}
    prev = None
    start_index = 0

    for i, element in enumerate(read):
        if element != prev:
            if prev is not None:
                collapsed_array.append(prev)
                count_dict[prev] = count_dict.get(prev, 0) + 1
                indices_dict[prev] = indices_dict.get(prev, []) + [(start_index, i - 1)]
            prev = element
            start_index = i

    if prev is not None:
        collapsed_array.append(prev)
        count_dict[prev] = count_dict.get(prev, 0) + 1
        indices_dict[prev] = indices_dict.get(prev, []) + [(start_index, len(read) - 1)]
    
    return collapsed_array, count_dict, indices_dict

############ check if elements are in expected order ##############

def match_order(array, target):
    """Check if array matches the target in different slices."""
    return(
        array == target or
        array[1:] == target or
        array[1:-1] == target or
        array[:-1] == target
    )

def check_order(collapsed_array, count_dict, expected_order):

    order_match_w_polyA = match_order(collapsed_array, expected_order)
    order_match_w_polyA_rev = match_order(collapsed_array, expected_order[::-1])

    polyA = "polyA" if "polyA" in expected_order else "polyT"

    expected_order_wo_polyA = [item for item in expected_order if item != polyA]
    order_match_wo_polyA = match_order(collapsed_array, expected_order_wo_polyA)
    order_match_w0_polyA_rev = match_order(collapsed_array, expected_order_wo_polyA[::-1])

    order_match = False
    order = ""

    reasons = "NA"

    if order_match_w_polyA:
        order_match = True
        order = "+"
    if order_match_wo_polyA:
        order_match = True
        order = "+"
        reasons = f'mi {polyA}'
    if order_match_w_polyA_rev:
        order_match = True
        order = "-"
    if order_match_w0_polyA_rev:
        order_match = True
        order = "-"
        reasons = f'mi {polyA}'

    if order_match:
        for key, count in count_dict.items():
            if key != "cDNA" and count > 1:
                order_match = False

    for element in expected_order:
        if not order_match:
            if element not in count_dict:
                reasons = reasons if reasons != "NA" else ""
                if reasons == "":
                    reasons += f"mi {element}"
                else:
                    reasons += f", mi {element}"
                # reasons += f", missing {element}"
            elif count_dict[element] > 1 and element != "cDNA":
                reasons = reasons if reasons != "NA" else ""
                if reasons == "":
                    reasons += f"mu {element}"
                else:
                    reasons += f", mu {element}"
            
    if not order_match and reasons == "NA":
        reasons = "unexpected order"
    
    return order_match, order, reasons

############## process full-length reads ############

def process_full_len_reads(data, barcodes, label_binarizer, model_path_w_CRF):
    
    read, prediction, read_length, seq_order = data
    
    if model_path_w_CRF:
        prediction = np.asarray(prediction)
        if prediction.ndim == 1:
            prediction = prediction[np.newaxis, :] 
            # decoded_prediction = label_binarizer.inverse_transform(prediction)[0]
            decoded_prediction = label_binarizer.classes_[prediction[0] if prediction.ndim == 2 else prediction]
    else:
        decoded_prediction = label_binarizer.inverse_transform(prediction)

    collapsed_array, count_dict, indices_dict = collapse_labels(decoded_prediction, read_length)
    order_match, order, reasons = check_order(collapsed_array, count_dict, seq_order)
    
    annotations = {element: {'Starts': [], 'Ends': [], 'Sequences': []} for element in seq_order}
    annotations['random_s'] = {'Starts': [], 'Ends': [], 'Sequences': []}
    annotations['random_e'] = {'Starts': [], 'Ends': [], 'Sequences': []}
    annotations['read'] = read[0:read_length]
    
    for element in indices_dict:
        for coordinates in indices_dict[element]:
            start, end = coordinates
            annotations[element]['Starts'].append(start)
            annotations[element]['Ends'].append(end)
    if order_match:
        if len(annotations["cDNA"]['Starts']) == 2:
            if order == "+" and collapsed_array[0] == "cDNA":
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]
            elif order == "+" and collapsed_array[-1] == "cDNA":
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][1])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][1])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][0]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][0]]
            elif order == "-" and collapsed_array[-1] == "cDNA":
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][1])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][1])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][0]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][0]]
            elif order == "-" and collapsed_array[0] == "cDNA":
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]

        if len(annotations["cDNA"]['Starts']) == 3:
            if order == "+":
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][2])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][2])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]
            else:
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][2])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][2])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]
            
    annotations["architecture"] = "valid" if order_match else "invalid"
    annotations['read_length'] = str(read_length)
    annotations['orientation'] = order
    annotations["reason"] = reasons
    
    if annotations["architecture"] == "valid":
        for barcode in barcodes:
            annotations[barcode]["Sequences"] = [read[int(annotations[barcode]['Starts'][0]):int(annotations[barcode]['Ends'][0]) + 1]]

    return annotations

def extract_annotated_full_length_seqs(new_data, predictions, model_path_w_CRF, read_lengths, label_binarizer, seq_order, barcodes, n_jobs):
    
    data = [(new_data[i], predictions[i], read_lengths[i], seq_order) for i in range(len(new_data))]

    with mp.Pool(processes=n_jobs) as pool:
        annotated_data = pool.starmap(process_full_len_reads, [(d, barcodes, label_binarizer, model_path_w_CRF) for d in data])
    return annotated_data
