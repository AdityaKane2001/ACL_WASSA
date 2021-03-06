<div align="center">
  
# Transformer based ensemble for emotion detection
[Aditya Kane](https://github.com/AdityaKane2001), [Shantanu Patankar](https://github.com/shantypat), [Sahil Khose](https://github.com/sahilkhose), [Neeraja Kirtane](https://github.com/neeraja1504)
</div>

- Official implementation of https://aclanthology.org/2022.wassa-1.25/. <br>
- Our WandB project is available [here](https://wandb.ai/acl_wassa_pictxmanipal/acl_wassa) 

--------------------------------------------------------------------------------------------
## Abstract
Detecting emotions in languages is important to accomplish a complete interaction between humans and machines. This paper describes our contribution to the WASSA 2022 shared task which handles this crucial task of emotion detection. We have to identify the following emotions: sadness, surprise, neutral, anger, fear, disgust, joy based on a given essay text. We are using an ensemble of ELECTRA and BERT models to tackle this problem achieving an F1 score of 62.76%. Our [codebase](https://github.com/AdityaKane2001/ACL_WASSA) and our [WandB project](https://wandb.ai/acl_wassa_pictxmanipal/acl_wassa) is available.

--------------------------------------------------------------------------------------------
## :books: Citation
If you find our paper useful in your research, please consider citing:
```
@inproceedings{kane-etal-2022-transformer,
    title = "Transformer based ensemble for emotion detection",
    author = "Kane, Aditya  and
      Patankar, Shantanu  and
      Khose, Sahil  and
      Kirtane, Neeraja",
    booktitle = "Proceedings of the 12th Workshop on Computational Approaches to Subjectivity, Sentiment {\&} Social Media Analysis",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wassa-1.25",
    pages = "250--254",
    abstract = "Detecting emotions in languages is important to accomplish a complete interaction between humans and machines. This paper describes our contribution to the WASSA 2022 shared task which handles this crucial task of emotion detection. We have to identify the following emotions: sadness, surprise, neutral, anger, fear, disgust, joy based on a given essay text. We are using an ensemble of ELECTRA and BERT models to tackle this problem achieving an F1 score of 62.76{\%}. Our codebase (https://bit.ly/WASSA{\_}shared{\_}task) and our WandB project (https://wandb.ai/acl{\_}wassa{\_}pictxmanipal/acl{\_}wassa) is publicly available.",
}
```
