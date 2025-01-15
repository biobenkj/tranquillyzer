# import pandas as pd
# from .barcode_correction import levenshtein_distance
# from .barcode_correction import correct_barcode
# import multiprocessing as mp

# ################ collapse labels into order ################

# def collapse_labels(arr, read_length):
#     # if not arr:
#     #     return [], {}, False

#     read = arr[0:read_length]

#     collapsed_array = []
#     count_dict = {}
#     indices_dict = {}
#     prev = None
#     start_index = 0

#     for i, element in enumerate(read):
#         if element != prev:
#             if prev is not None:
#                 collapsed_array.append(prev)
#                 count_dict[prev] = count_dict.get(prev, 0) + 1
#                 indices_dict[prev] = indices_dict.get(prev, []) + [(start_index, i - 1)]
#             prev = element
#             start_index = i

#     if prev is not None:
#         collapsed_array.append(prev)
#         count_dict[prev] = count_dict.get(prev, 0) + 1
#         indices_dict[prev] = indices_dict.get(prev, []) + [(start_index, len(read) - 1)]
    
#     return collapsed_array, count_dict, indices_dict

# ############ check if elements are in expected order ##############

# def check_order(collapsed_array, count_dict, expected_order):
   
#     order_match_1 = collapsed_array == expected_order
#     order_match_2 = collapsed_array[1:len(collapsed_array)] == expected_order
#     order_match_3 = collapsed_array[1:len(collapsed_array)-1] == expected_order
#     order_match_4 = collapsed_array[0:len(collapsed_array)-1] == expected_order

#     expected_order_rev = expected_order[::-1]

#     order_match_5 = collapsed_array == expected_order_rev
#     order_match_6 = collapsed_array[1:len(collapsed_array)] == expected_order_rev
#     order_match_7 = collapsed_array[1:len(collapsed_array)-1] == expected_order_rev
#     order_match_8 = collapsed_array[0:len(collapsed_array)-1] == expected_order_rev

#     order_match = False
#     order = ""

#     if order_match_1 or order_match_2 or order_match_3 or order_match_4: 
#         order_match = True
#         order = "forward"
#     if order_match_5 or order_match_6 or order_match_7 or order_match_8:
#         order_match = True
#         order = "reverse"

#     reasons = "NA"

#     for element in expected_order:
#         if order_match == False:
#             if element not in count_dict:
#                 if reasons == "NA":
#                     reasons = "missing " + element
#                 else:
#                     reasons = reasons + ", missing " + element
#             elif count_dict[element] > 1 and element != "cDNA":
#                 if reasons == "NA":
#                     reasons = "multiple " + element
#                 else:
#                     reasons = reasons + ", multiple " + element
            
#     if order_match == False and reasons == "NA":
#         reasons = "unexpected order"
    
#     return order_match, order, reasons

# # def modify_indices_dict(order_match, reasons, collapsed_array, indices_dict):
# #     if order_match or reasons == "multiple polyT":
# #         if collapsed_array[0] == 'cDNA':
# #             collapsed_array[0] = "random_s"
# #             indices_dict['random_s'] = indices_dict['cDNA'][0]
# #             indices_dict['cDNA'] = indices_dict['cDNA'][1:]
            
# #         if collapsed_array[len(collapsed_array)-1] == 'cDNA':
# #             collapsed_array[len(collapsed_array)-1] = "random_e"
# #             indices_dict['random_e'] = indices_dict['cDNA'][len(indices_dict['cDNA'])-1]
# #             indices_dict['cDNA'] = indices_dict['cDNA'][:len(indices_dict['cDNA'])-1]
    
# ############ process read ends ##############

# def extract_annotated_seq_ends(new_data, predictions, label_binarizer, actual_lengths, n, seq_order):
#     decoded_predictions = [label_binarizer.inverse_transform(seq) for seq in predictions]

#     annotated_data = []
    
#     for i in range(0, len(new_data), 2):
#         read_1, read_2 = new_data[i], new_data[i + 1]
#         predictions_1, predictions_2 = decoded_predictions[i], decoded_predictions[i + 1]
#         actual_length = actual_lengths[int(i/2)]

#         fe_collapsed_array, fe_count_dict, fe_indices_dict = collapse_labels(predictions_1, n) # first end
#         se_collapsed_array, se_count_dict, se_indices_dict = collapse_labels(predictions_2, n) # second end

#         if (fe_collapsed_array[len(fe_collapsed_array)-1] == "cDNA") and (se_collapsed_array[0] == "cDNA"):
#             collapsed_array = fe_collapsed_array[0:len(fe_collapsed_array)-1] + se_collapsed_array
#         else:
#             collapsed_array = fe_collapsed_array + se_collapsed_array # combined order

#         count_dict = fe_count_dict # combined count

#         for element in se_count_dict:
#             if element not in count_dict:
#                 count_dict[element] = se_count_dict[element]
#             else:
#                 count_dict[element] = count_dict[element] + se_count_dict[element]

#         order_match, reasons = check_order(collapsed_array, count_dict, seq_order)

#         annotations = {}
#         for element in seq_order:
#             annotations[element] = {'Starts': [], 'Ends': [], 'Sequences': []}
        
#         annotations['random_s'] = {'Starts': [], 'Ends': [], 'Sequences': []}
#         annotations['random_e'] = {'Starts': [], 'Ends': [], 'Sequences': []}

#         for element in fe_indices_dict:
#             for coordinates in fe_indices_dict[element]:
#                 start = coordinates[0]
#                 end = coordinates[1]
#                 if element == "cDNA":
#                     if end == (n - 1):
#                         annotations[element]['Starts'].append(start)
#                     else:
#                         annotations[element]['Starts'].append(start)
#                         annotations[element]['Ends'].append(end)
#                 elif element != "cDNA":
#                     annotations[element]['Starts'].append(start)
#                     annotations[element]['Ends'].append(end)
#                     annotations[element]['Sequences'].append(read_1[start: (end + 1)])

#         for element in se_indices_dict:
#             for coordinates in se_indices_dict[element]:
#                 start = coordinates[0]
#                 end = coordinates[1]
#                 if element == "cDNA":
#                     if start == 0:
#                         annotations[element]['Ends'].append(actual_length - n + end)
#                     else:
#                         annotations[element]['Starts'].append(actual_length - n + start)
#                         annotations[element]['Ends'].append(actual_length - n + end)
#                 elif element != "cDNA":
#                     annotations[element]['Starts'].append(start)
#                     annotations[element]['Ends'].append(end)
#                     annotations[element]['Sequences'].append(read_2[start: (end + 1)])      

#         if order_match or (order_match == False and reasons == "multiple polyT"):
#             annotations['random_s']['Starts'] = [annotations['cDNA']['Starts'][0]]
#             annotations['random_s']['Ends'] = [annotations['cDNA']['Ends'][0]]

#             annotations['random_e']['Starts'] = [annotations['cDNA']['Starts'][-1]]
#             annotations['random_e']['Ends'] = [annotations['cDNA']['Ends'][-1]]

#             annotations['cDNA']['Starts'] = annotations['cDNA']['Starts'][1:-1]
#             annotations['cDNA']['Ends'] = annotations['cDNA']['Ends'][1:-1]

#         if order_match:
#             annotations["architecture"] = "valid"
#         else:
#             annotations["architecture"] = "invalid"
        
#         annotations["reason"] = reasons
#         annotated_data.append(annotations)

#     return annotated_data

# ############## process full-length reads ############

# def process_full_len_reads(data):

#     read, prediction, read_length, seq_order = data
#     collapsed_array, count_dict, indices_dict = collapse_labels(prediction, read_length)
#     order_match, order, reasons = check_order(collapsed_array, count_dict, seq_order)
    
#     annotations = {}
    
#     for element in seq_order:
#         annotations[element] = {'Starts': [], 'Ends': [], 'Sequences': []}
        
#     annotations['random_s'] = {'Starts': [], 'Ends': [], 'Sequences': []}
#     annotations['random_e'] = {'Starts': [], 'Ends': [], 'Sequences': []}
    
#     for element in indices_dict:
#         for coordinates in indices_dict[element]:
#             start = coordinates[0]
#             end = coordinates[1]
#             annotations[element]['Starts'].append(start)
#             annotations[element]['Ends'].append(end)
#             if element != "cDNA":
#                 annotations[element]['Sequences'].append(read[start: (end + 1)])
                
#     if order_match:
#         # print(len(annotations["cDNA"]['Starts']))
#         if (len(annotations["cDNA"]['Starts']) == 2) and (order == "forward"):
#             if (collapsed_array[0] == "cDNA"):
#                 annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][0])
#                 annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][0])
#                 annotations['cDNA']['Starts'] == [annotations['cDNA']['Starts'][1]]
#                 annotations['cDNA']['Ends'] == [annotations['cDNA']['Ends'][1]]
#             elif (collapsed_array[-1] == "cDNA"):
#                 annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][1])
#                 annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][1])
#                 annotations['cDNA']['Starts'] == [annotations['cDNA']['Starts'][0]]
#                 annotations['cDNA']['Ends'] == [annotations['cDNA']['Ends'][0]]
        
#         if (len(annotations["cDNA"]['Starts']) == 2) and (order == "reverse"):
#             if (collapsed_array[0] == "cDNA"):
#                 annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][0])
#                 annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][0])
#                 annotations['cDNA']['Starts'] == [annotations['cDNA']['Starts'][1]]
#                 annotations['cDNA']['Ends'] == [annotations['cDNA']['Ends'][1]]
#             elif (collapsed_array[-1] == "cDNA"):
#                 annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][1])
#                 annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][1])
#                 annotations['cDNA']['Starts'] == [annotations['cDNA']['Starts'][0]]
#                 annotations['cDNA']['Ends'] == [annotations['cDNA']['Ends'][0]]

#         if (len(annotations["cDNA"]['Starts']) == 3) and (order == "forward"):
#             annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][0])
#             annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][0])
#             annotations['cDNA']['Starts'] == [annotations['cDNA']['Starts'][1]]
#             annotations['cDNA']['Ends'] == [annotations['cDNA']['Ends'][1]]
#             annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][2])
#             annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][2])

#         if (len(annotations["cDNA"]['Starts']) == 3) and (order == "reverse"):
#             annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][0])
#             annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][0])
#             annotations['cDNA']['Starts'] == [annotations['cDNA']['Starts'][1]]
#             annotations['cDNA']['Ends'] == [annotations['cDNA']['Ends'][1]]
#             annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][2])
#             annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][2])

#     if order_match:
#         annotations["architecture"] = "valid"
#     else:
#         annotations["architecture"] = "invalid"

#     annotations['read_length'] = str(read_length)
#     annotations['orientation'] = order
        
#     annotations["reason"] = reasons
#     return annotations

# def extract_annotated_full_length_seqs(new_data, predictions, read_lengths, label_binarizer, seq_order, n_jobs=4):
#     decoded_predictions = [label_binarizer.inverse_transform(seq) for seq in predictions]
    
#     # Prepare data for parallel processing
#     data = [(new_data[i], decoded_predictions[i], read_lengths[i], seq_order) for i in range(len(new_data))]
    
#     # Determine chunk size and number of chunks
#     chunk_size = len(data) // n_jobs
#     remainder = len(data) % n_jobs
    
#     # Split data into equal chunks
#     chunks = [data[i:i + chunk_size] for i in range(0, len(data) - remainder, chunk_size)]
    
#     # Add the remaining reads to the last chunk
#     if remainder > 0:
#         chunks[-1].extend(data[-remainder:])
    
#     # Create a Pool of workers
#     with mp.Pool(processes=n_jobs) as pool:
#         # Parallel map using chunks
#         annotated_data = pool.map(process_full_len_reads, data)
    
#     return annotated_data

import polars as pl
from .barcode_correction import levenshtein_distance, correct_barcode
import multiprocessing as mp

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

def check_order(collapsed_array, count_dict, expected_order):

    order_match_1 = collapsed_array == expected_order
    order_match_2 = collapsed_array[1:len(collapsed_array)] == expected_order
    order_match_3 = collapsed_array[1:len(collapsed_array)-1] == expected_order
    order_match_4 = collapsed_array[0:len(collapsed_array)-1] == expected_order

    expected_order_rev = expected_order[::-1]

    order_match_5 = collapsed_array == expected_order_rev
    order_match_6 = collapsed_array[1:len(collapsed_array)] == expected_order_rev
    order_match_7 = collapsed_array[1:len(collapsed_array)-1] == expected_order_rev
    order_match_8 = collapsed_array[0:len(collapsed_array)-1] == expected_order_rev

    order_match = False
    order = ""

    if order_match_1 or order_match_2 or order_match_3 or order_match_4: 
        order_match = True
        order = "forward"
    if order_match_5 or order_match_6 or order_match_7 or order_match_8:
        order_match = True
        order = "reverse"

    if order_match:
        for key, count in count_dict.items():
            if key != "cDNA" and count > 1:
                order_match = False

    reasons = "NA"

    for element in expected_order:
        if not order_match:
            if element not in count_dict:
                reasons = reasons if reasons != "NA" else "missing " + element
                reasons += f", missing {element}"
            elif count_dict[element] > 1 and element != "cDNA":
                reasons = reasons if reasons != "NA" else ""
                if reasons == "":
                    reasons += f"multiple {element}"
                else:
                    reasons += f", multiple {element}"
            
    if not order_match and reasons == "NA":
        reasons = "unexpected order"
    
    return order_match, order, reasons

############ process read ends ##############

def extract_annotated_seq_ends(new_data, predictions, label_binarizer, actual_lengths, n, seq_order):
    decoded_predictions = [label_binarizer.inverse_transform(seq) for seq in predictions]

    annotated_data = []
    
    for i in range(0, len(new_data), 2):
        read_1, read_2 = new_data[i], new_data[i + 1]
        predictions_1, predictions_2 = decoded_predictions[i], decoded_predictions[i + 1]
        actual_length = actual_lengths[int(i/2)]

        fe_collapsed_array, fe_count_dict, fe_indices_dict = collapse_labels(predictions_1, n) # first end
        se_collapsed_array, se_count_dict, se_indices_dict = collapse_labels(predictions_2, n) # second end

        if (fe_collapsed_array[-1] == "cDNA") and (se_collapsed_array[0] == "cDNA"):
            collapsed_array = fe_collapsed_array[:-1] + se_collapsed_array
        else:
            collapsed_array = fe_collapsed_array + se_collapsed_array

        count_dict = fe_count_dict

        for element in se_count_dict:
            count_dict[element] = count_dict.get(element, 0) + se_count_dict[element]

        order_match, order, reasons = check_order(collapsed_array, count_dict, seq_order)

        annotations = {element: {'Starts': [], 'Ends': [], 'Sequences': []} for element in seq_order}
        annotations['random_s'] = {'Starts': [], 'Ends': [], 'Sequences': []}
        annotations['random_e'] = {'Starts': [], 'Ends': [], 'Sequences': []}

        for element in fe_indices_dict:
            for coordinates in fe_indices_dict[element]:
                start, end = coordinates
                if element == "cDNA" and end == (n - 1):
                    annotations[element]['Starts'].append(start)
                else:
                    annotations[element]['Starts'].append(start)
                    annotations[element]['Ends'].append(end)
                if element != "cDNA":
                    annotations[element]['Sequences'].append(read_1[start: (end + 1)])

        for element in se_indices_dict:
            for coordinates in se_indices_dict[element]:
                start, end = coordinates
                if element == "cDNA":
                    if start == 0:
                        annotations[element]['Ends'].append(actual_length - n + end)
                    else:
                        annotations[element]['Starts'].append(actual_length - n + start)
                        annotations[element]['Ends'].append(actual_length - n + end)
                else:
                    annotations[element]['Starts'].append(actual_length - n + start)
                    annotations[element]['Ends'].append(actual_length - n + end)
                    annotations[element]['Sequences'].append(read_2[start: (end + 1)])      

        if order_match or (not order_match and reasons == "multiple polyT"):
            annotations['random_s']['Starts'] = [annotations['cDNA']['Starts'][0]]
            annotations['random_s']['Ends'] = [annotations['cDNA']['Ends'][0]]

            annotations['random_e']['Starts'] = [annotations['cDNA']['Starts'][-1]]
            annotations['random_e']['Ends'] = [annotations['cDNA']['Ends'][-1]]

            annotations['cDNA']['Starts'] = annotations['cDNA']['Starts'][1:-1]
            annotations['cDNA']['Ends'] = annotations['cDNA']['Ends'][1:-1]

        annotations["architecture"] = "valid" if order_match else "invalid"
        annotations["reason"] = reasons
        annotated_data.append(annotations)

    return annotated_data

############## process full-length reads ############

def process_full_len_reads(data, barcodes):

    read, prediction, read_length, seq_order = data
    collapsed_array, count_dict, indices_dict = collapse_labels(prediction, read_length)
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
            if order == "forward" and collapsed_array[0] == "cDNA":
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]
            elif order == "forward" and collapsed_array[-1] == "cDNA":
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][1])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][1])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][0]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][0]]
            elif order == "reverse" and collapsed_array[-1] == "cDNA":
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][1])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][1])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][0]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][0]]
            elif order == "reverse" and collapsed_array[0] == "cDNA":
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]

        if len(annotations["cDNA"]['Starts']) == 3:
            if order == "forward":
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

def extract_annotated_full_length_seqs(new_data, predictions, read_lengths, label_binarizer, seq_order, barcodes, n_jobs=4):
    decoded_predictions = [label_binarizer.inverse_transform(seq) for seq in predictions]
    
    # Prepare data for parallel processing
    data = [(new_data[i], decoded_predictions[i], read_lengths[i], seq_order) for i in range(len(new_data))]
    
    # Determine chunk size and number of chunks
    chunk_size = len(data) // n_jobs
    remainder = len(data) % n_jobs
    
    # Split data into equal chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data) - remainder, chunk_size)]
    
    if remainder > 0:
        chunks[-1].extend(data[-remainder:])
    
    # annotated_data = process_full_len_reads(data, barcodes)

    # Create a Pool of workers
    with mp.Pool(processes=n_jobs) as pool:
        annotated_data = pool.starmap(process_full_len_reads, [(d, barcodes) for d in data])
    
    return annotated_data