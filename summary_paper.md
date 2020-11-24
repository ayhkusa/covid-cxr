# Section II. Summary Paper

## Evolution of Machine Learning

Recent advances in AI and ML have propelled the need to explain the models predictions. As more industries adopting AI/ML for all kinds of applications like forecasting sales, ranking products, rating employees, etc. We need better understanding how these complex AI/ML models behave when it achieves positive predictions as well as failed predictions.

## Who Needs to Know

Data Scientists – These individuals who works and creates ML models. They are the first hands to experience and to reveal the underlining behaviors of the models. Having an explainable model would add value during ML development and troubleshooting. Especially valuable with image type of classifications to aid identify features contributed to predictions.

Domain Experts – These are the individual with specialization in a particular field. Such as doctors that looks at a patient’s x-ray to decide if the case of ailment is positive or negative. Their extensive training and experience would complement the ML models. It is especially critical that when there is a disagreement between the model and human judgement. We need to know how the model made the conclusion and factors it used for the decision.

Business Stakeholders – These individuals would make the decision if the ML program would have the necessary funding. The ML model must provide explanations that are human friendly. Buy-ins from the stakeholders is critical to proliferation the ML program.

End Users – These are the people that will use the ML program on a daily basis. They are the eyes and guards of the model’s behavior. The human intuition coupled with explainable ML program would build trust and provide additional benefits in decision making.

## Why ML Explainability

ML programs interpret the data with algorithms to predict an event without knowing why. Therefore it is important for ML creators and users to understand the predictions in human terms with transparency.  

## What Explainable Models are in Use:

Integrated Gradients is recommended for neural networks especially for large input feature such as images. 

XRAI is recommended for image models where it's desirable to localize attributions at the region vs. pixel level. 

Shapley is recommended for tabular and non-differentiable models, which is the case in AutoML Tables models consisting of meta-ensembles of trees and neural networks.

LIME is a ML model agnostic method by testing the ML model results when certain aspects are changed within the model. 

## What to Explain

• Bias: This concern is high on the list of explainability. Does the ML model lean towards a specify data pattern creating unfairness? Ability to recognize this behavior that provides an opportunity for corrective action.

• Correctness: Did the model use arrive at similar features to prediction the results as humans do? Or something totally different to how human would interprets the data.

• Human comprehensibility: Ability to convey the results that is equally understandable in language of the ML end user. As if one human to another human of equal skills.

• Improvement: Can the ML model explain the result that may be overlooked by domain experts? Provides an opportuntity to further enhance the ML model in iterations.

• Robustness: Would the model detect interferences that may be unintentional or intentional. Ability to hightlight the differences and reveal the source of the error.

• Transferability: Is the explainability scope limited to single application or shareable across multiple domains. 

## How to Explain

• Text explanations by spoken lanugage relevant to the subject domain.

• Visual explanation by highlighting or overlaying images with additional visual aids.

• Local explanations focus on the area that contributed to the prediction vs trying to explain everything.

• Explanations by example to show similarities with another example.

## Conclusion

Explainability is one facet of ML journey. It is critical to ML practitioners and users to establish a trust in using ML programs. Explainability must be conveyed in human terms that doesn't require an user to have extensive ML training. The SHAP explanation is one of the latest model and adoptable across various ML techinques. SHAP has limitations, such as unrepresentative features and feature independence. But knowing these limitations are valuable to ensure there are balance of features and robust associations of features in the data source, respectively. Forming a feedback loop to iterate improvements in ML models.


