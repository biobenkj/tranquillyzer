import os

def trained_models():
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    models_dir = os.path.join(base_dir, "..", "models")   
    models_dir = os.path.abspath(models_dir)

    utils_dir = os.path.join(base_dir, "..", "utils")
    utils_dir = os.path.abspath(utils_dir)
    
    try:
        # Check if the directory exists
        if not os.path.isdir(models_dir):
            print(f"The directory '{models_dir}' does not exist.")
            return

        print("\n ~~~~~~~~~~~~~~~~ CURRENTLY AVAILABLE TRAINED MODELS ~~~~~~~~~~~~~~~~\n")
        # Iterate over all files in the directory
        for file_name in os.listdir(models_dir):
            # Check if the file has a .h5 extension
            seq_order, sequences, barcodes, UMIs = seq_orders(os.path.join(utils_dir, "seq_orders.tsv"), file_name[:-3])
            if file_name.endswith('.h5'):
                # Print the file name without the .h5 extension
                print("-- " + file_name[:-3] + " \t layout ==> " + ','.join(map(str, seq_order)) + " \n\t\t\t sequences ==> " + ','.join(map(str, sequences)) + "\n")
        print("\n")

    except Exception as e:
        print(f"An error occurred: {e}")

def seq_orders(file_path, model):
    try:
        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"The file '{file_path}' does not exist.")
            return

        sequence_order = []
        sequences = []
        barcodes = []
        UMIs = []

        # Open the file and read lines
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line by commas
                fields = line.strip().split("\t")

                model_name = fields[0].rstrip()
                sequence_order = fields[1][1:-1].rstrip().split(',')
                sequences = fields[2][1:-1].rstrip().split(',')

                barcodes = fields[3]
                if (barcodes.startswith("'") and barcodes.endswith("'")) or (barcodes.startswith("\"") and barcodes.endswith("\"")):
                    barcodes = barcodes[1:-1] 
                barcodes = barcodes.rstrip().split(',')
                
                UMIs = fields[4]
                if (UMIs.startswith("'") and UMIs.endswith("'")) or (UMIs.startswith("\"") and UMIs.endswith("\"")):
                    UMIs = UMIs[1:-1] 
                UMIs = UMIs.rstrip().split(',')
                
                # The first part is the model name, the rest are sequences
                if model_name == model:
                    sequence_order = sequence_order
                    sequences = sequences
                    barcodes = barcodes
                    UMIs = UMIs
                    break
        
        return sequence_order, sequences, barcodes, UMIs
    except Exception as e:
        print(f"An error occurred: {e}")

import os

def training_seq_orders(file_path, model):
    try:
        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"The file '{file_path}' does not exist.")
            return

        sequence_order = []
        sequences = []
        barcodes = []
        UMIs = []
        training_seq_orders = []  # New list to store sequence orders

        # Open the file and read lines
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line by tab
                fields = line.strip().split("\t")

                if len(fields) < 5:  # Ensure at least 5 columns exist
                    continue

                model_name = fields[0].strip()
                sequence_order = [seg.strip() for seg in fields[1][1:-1].split(',')]
                sequences = [seg.strip() for seg in fields[2][1:-1].split(',')]

                # Parse barcodes
                barcodes = fields[3].strip()
                if (barcodes.startswith("'") and barcodes.endswith("'")) or (barcodes.startswith("\"") and barcodes.endswith("\"")):
                    barcodes = barcodes[1:-1]
                barcodes = [b.strip() for b in barcodes.split(',')]

                # Parse UMIs
                UMIs = fields[4].strip()
                if (UMIs.startswith("'") and UMIs.endswith("'")) or (UMIs.startswith("\"") and UMIs.endswith("\"")):
                    UMIs = UMIs[1:-1]
                UMIs = [u.strip() for u in UMIs.split(',')]

                # Check if a 6th column exists
                if len(fields) > 5:
                    training_order_str = fields[5].strip()
                    if training_order_str:
                        training_seq_orders = [
                            [seg.strip() for seg in order.split(',')] for order in training_order_str.split(':')
                        ]
                    # Return all five values if column 6 is present
                    if model_name == model:
                        return sequence_order, sequences, barcodes, UMIs, training_seq_orders
                
                # Return only the first four values if column 6 is missing
                if model_name == model:
                    return sequence_order, sequences, barcodes, UMIs, []

        # Default return values (if no matching model is found)
        if training_seq_orders:
            return sequence_order, sequences, barcodes, UMIs, training_seq_orders
        else:
            return sequence_order, sequences, barcodes, UMIs, []

    except Exception as e:
        print(f"An error occurred: {e}")


