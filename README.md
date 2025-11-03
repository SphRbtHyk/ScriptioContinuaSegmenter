# Hierarchical transformers for the segmentation of Greek and Latin texts

This repository implements a powerful two-stage text segmentation system that combines **character-level classification** with **linguistic validation** to deliver highly accurate segmentation of texts written in Scriptio Continua.

## Overview 🧐

Segmenting ancient, noisy, or unsegmented text presents unique challenges. Traditional approaches often rely solely on classification models, which can produce linguistically incoherent results. Our **HierarchicalSegmenter** solves this by implementing a two-stage process:

1. 🎯 **Character-Level Classification**: Uses beam search to identify the top-k most probable segmentation points
2. 🧠 **Linguistic Validation**: Employs transformer perplexity scoring to select the most coherent segmentation

This approach is designed for ancient manuscripts, historical documents, and texts without clear word boundaries. 📜

## Installation ⚡

After cloning the repository, run:
```bash
pip install .
```

## Quick Start 🚀

```python
from sc_segmenter.segmenters.hierarchical_segmenter import HierarchicalSegmenter

# Initialize with trained models
# Here we used default facebook/xglm-564M
# And an included demo model trained for latin
from sc_segmenter.segmenters.hierarchical_segmenter import HierarchicalSegmenter

# Initialize with your trained models
segmenter = HierarchicalSegmenter(
    autoregressive_model_path="facebook/xglm-564M",
    character_model_path="SphRbtHyk/demo_latin_character_segmenter",
    # Demonstration trained Latin classifier 
    beam_width=3
)

# Segment text without spaces
text = "inprincipioeratverbum"
result = segmenter.segment(text)
print(f"Segmented: {result}")

# Output: Segmented: ['in', 'principio', 'erat', 'verbum']
```

## How It Works 🔧

### Stage 1: Beam Search Segmentation
- A character-level classification model analyzes the input text
- Beam search explores multiple segmentation possibilities simultaneously  
- Returns the top-k most probable segmentation sequences based on classification confidence

### Stage 2: Perplexity-Based Selection
- Each candidate segmentation is evaluated by a language model
- The model computes perplexity scores (lower = more linguistically coherent)
- The segmentation with the lowest perplexity is selected as the final output


## Citation 📚

PAPER UNDER REVIEW

## License

MIT License
