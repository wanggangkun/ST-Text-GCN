## ST-Text-GCN
Code for paper "Self-training Method Based on GCN for Semi-supervised Short Text Classification"
### Reproducing Results
Take the Ohsumed dataset as an example, other datasets are similar, number of round is 2.
```bash
cd data/Ohsumed/raw
python data_process.py
cd ../../../code
python build_graph.py --dataset Ohsumed --build_time 1
python train.py --dataset Ohsumed
python build_graph.py --dataset Ohsumed --build_time 2
python train.py --dataset Ohsumed
```
