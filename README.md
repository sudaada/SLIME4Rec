# SLIME4Rec
Code for ICDE 2023 paper, **Contrastive Enhanced Slide Filter Mixer for Sequential Recommendation**.

# Requriements
- Install Python, Pytorch(>=1.8). We use Python 3.8, Pytorch 1.8.
- If you plan to use GPU computation, install CUDA.

# Usage
Download datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.
- [Amazon_Beauty](https://drive.google.com/file/d/16BDe6IT0mQeBK519tEPSgofdRF2KwgPV/view?usp=share_link)
- [Amazon_Clothing_Shoes_and_Jewelry](https://drive.google.com/file/d/1vH8Wn792iR69WK4iqjRrCGWFTi72v2FE/view?usp=share_link)
- [Amazon_Sports_and_Outdoors](https://drive.google.com/file/d/1VkC55X9NTOsnwApFYEZvTQ0807bLUzlZ/view?usp=share_link)
- [ml-1m](https://drive.google.com/file/d/1sqgFpwHNWNPaMlVFHbQQXIRaAN9i3KUJ/view?usp=share_link)
- [yelp](https://drive.google.com/file/d/1x5I2wHvKf2C4KxtczGHLNvofHX_G5fS3/view?usp=share_link)

```
$ tree
.
├── Amazon_Beauty
│   ├── Amazon_Beauty.inter
│   └── Amazon_Beauty.item
├── Amazon_Clothing_Shoes_and_Jewelry
│   ├── Amazon_Clothing_Shoes_and_Jewelry.inter
│   └── Amazon_Clothing_Shoes_and_Jewelry.item
├── Amazon_Sports_and_Outdoors
│   ├── Amazon_Sports_and_Outdoors.inter
│   └── Amazon_Sports_and_Outdoors.item
├── ml-1m
│   ├── ml-1m.inter
│   ├── ml-1m.item
│   ├── ml-1m.user
│   └── README.md
└── yelp
    ├── README.md
    ├── yelp.inter
    ├── yelp.item
    └── yelp.user
```
# Quick-Strat

Run `slime4rec.sh`.


# Acknowledgement
This repo is developed based on [RecBole](https://github.com/RUCAIBox/RecBole).
# SLIME4Rec
