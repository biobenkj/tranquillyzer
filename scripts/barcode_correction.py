def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # Initialize the distance matrix
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def correct_barcode(barcode, whitelist):
    """Correct the given barcode using the whitelist and determine closest matches."""
    min_distance = float('inf')
    closest_barcodes = []

    for valid_barcode in whitelist:
        distance = levenshtein_distance(barcode, valid_barcode)
        if distance < min_distance:
            min_distance = distance
            closest_barcodes = [valid_barcode]
        elif distance == min_distance:
            closest_barcodes.append(valid_barcode)
    
    if not closest_barcodes:
        return None, None, 'No match found'
    
    match_status = 'Single match' if len(closest_barcodes) == 1 else 'Multiple matches'
    return closest_barcodes, min_distance, match_status
