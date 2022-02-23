# When AI Difficulty is Easy: The Explanatory Power of Predicting IRT Difficulty
---

**Publication**: Fernando Mart&iacute;nez-Plumed,  David Castellano-Falcón, Carlos Monserrat, Jos&eacute; Hern&aacute;ndez-Orallo: [*"When AI Difficulty is Easy:
The Explanatory Power of Predicting IRT Difficulty"*](),  [36th AAAI Conference on Artificial Intelligence (AAAI 2022)](https://aaai.org/Conferences/AAAI-22/), Feb 22 - March 1, Vancouver, Canada, 2022.

**Description**: One of challenges of artificial intelligence as a whole is robustness. Many issues such as adversarial examples, out of  distribution performance, Clever Hans phenomena, and the wider areas of AI evaluation and explainable AI, have to do  with the following question: Did the system fail because it is  a hard instance or because something else? In this work we address this question with a generic method for estimating IRT-based instance difficulty for a wide range of AI domains covering several areas, from supervised feature-based classification to automated reasoning. We show how to estimate difficulty systematically using off-the-shelf machine learning regression models. We illustrate the usefulness of this estimation for a range of applications.


**Contribution**: We covers a range of problems in AI, derive their IRT difficulties, and train a regression model for each domain---a difficulty estimator---, which we evaluate systematically. For many domains, the estimates for IRT difficulty are very good, according to RMSE and Spearman correlation.  We illustrate the explanatory power of these difficulty models on a series of applications:


* **Explainable AI**: understanding what makes instances hard, or groups of instances (e.g., classes), and explaining whether an error is expected or unexpected.
* **Robust evaluation**: comparing systems using their characteristic curves. Systems that are reliable on all (or most) easy instances should be considered more robust. 
* **AI progress**: analysing whether the increase of performance has focused on the low-hanging easy instances or more complex instances through specialisation. %recognise the patterns that make instances hard  and address the difficulty gradient with better techniques.
* **Distribution changes and perturbations**:  a very capable system failing on a batch of very easy instances may suggest a distributional shift or an adversarial attack. The inverse phenomenon may signal a Clever Hans effect.

The main contributions of this paper are: (1) the first general methodology for training an estimator for IRT difficulties, (2) comprehensive empirical results showing the wide range of domains where it works, and (3) the evidence of its applicability as a powerful explanatory tool.
