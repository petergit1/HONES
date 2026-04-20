# GQA Dataset

## Description
GQA is a visual question answering dataset designed for real-world image understanding, with questions that involve visual, spatial, and compositional reasoning. In our experiments, GQA can be used as an out-of-distribution VQA benchmark.

## Download

The official GQA dataset can be downloaded from the GQA project website:

- Project page: https://cs.stanford.edu/people/dorarad/gqa/
- Dataset/download page: https://cs.stanford.edu/people/dorarad/gqa/download.html

After downloading, organize the files according to your own data directory. A typical structure can be:

```text
data/
  GQA/
    images/
    questions/
      train_balanced_questions.json
      val_balanced_questions.json
      testdev_balanced_questions.json
```


## Evaluation

For our setting, GQA is mainly used as a VQA-style evaluation dataset. The commonly reported metric is accuracy.

## Reference

Drew A. Hudson and Christopher D. Manning. **GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering.** CVPR 2019.
