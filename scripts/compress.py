from llmlingua import PromptCompressor
import pandas as pd
compressor = PromptCompressor()
df = pd.read_csv('dataset.csv', sep='|')
compressed_prompt = compressor.compress(
    df, 
    instruction="..."
)
