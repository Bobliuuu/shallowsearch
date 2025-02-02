import re

def extract_errors(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
    
    errors = []
    
    for i in range(len(lines)):
        if lines[i].strip() == "Prediction results:":
            block = lines[i:i+5] 
            
            for line in block:
                loss_match = re.search(r"loss=(True|([0-9]*\.?[0-9]+))", line)
                if loss_match:
                    loss_value = loss_match.group(1)
                    if loss_value == "True" or (loss_value.replace(".", "").isdigit() and float(loss_value) > 0.3):
                        errors.append("".join(block))
                        break 
    
    if errors:
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.writelines(errors)

if __name__ == "__main__":
    extract_errors("groq_log_reverse.txt", "error.txt")
