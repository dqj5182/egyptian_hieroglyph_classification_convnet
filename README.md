# Egyptian Hieroglyph Classification using ConvNet

This is a [PyTorch](https://pytorch.org/) implementation of the paper ["A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification" by Barucci et al](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9528382&tag=1).

Please download the dataset "EgyptianHieroglyphDataset_Original" at my [Google Drive](https://drive.google.com/drive/folders/1bhnMJ8NbCa-qw53EKy-olZp3cJKZU_jc?usp=sharing).

**Image** | ![alt text](/example/D21.png) | ![alt text](/example/E34.png) | ![alt text](/example/V31.png) 
------------ | ------------ | ------------- | -------------
**Gardener Label** | D21 | E34 | V31

Steps:
1. Download "EgyptianHieroglyphDataset_Original" dataset from my Google Drive
2. Install all the requirements for Python packages
3. Run main.py
4. The training will be starting right away!

Prior implementations:
1. [GlyphReader by Morris Franken](https://github.com/morrisfranken/glyphreader) which extracts features using Inception-v3 and classifies hieroglyphs using SVM.

TODO:
1. Implementation for Glyphnet
2. Implementation for Inception-v3
3. Implementation for Xception
