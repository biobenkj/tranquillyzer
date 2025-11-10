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
        strand = ""

        # Open the file and read lines
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line by tabs, removing extra quote characters at the same time
                fields = line.strip().replace("'", "").replace("\"", "").split("\t")

                # Check if desired model has been found
                # If so, process rest of the line
                model_name = fields[0].strip()
                if model_name == model:
                    sequence_order = fields[1].strip().split(',')
                    sequences = fields[2].strip().split(',')
                    barcodes = fields[3].strip().split(',')
                    UMIs = fields[4].strip().split(',')
                    strand = fields[5].strip()

                    break

                # Model name not found on this line, moving to the next one

        return sequence_order, sequences, barcodes, UMIs, strand
    except Exception as e:
        print(f"An error occurred: {e}")


