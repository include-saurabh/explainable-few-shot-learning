# Exploring Interpretable Few-Shot Learning: A Journey Through Attention and Prototypes

This project delves into the challenge of making Few-Shot Learning (FSL) models, particularly for image recognition, more interpretable. The exploration covers initial attempts using attention mechanisms with standard Prototypical Networks and culminates in the development of a novel hybrid model, HEPN, inspired by ProtoPNet, aiming to provide explanations based on learned prototypical parts.

## Table of Contents
1.  [Motivation: Why Explain AI Decisions?](#motivation-why-explain-ai-decisions)
2.  [The Challenge: Learning from Few Examples](#the-challenge-learning-from-few-examples)
3.  [Attempt 1: Post-Hoc Attention with ProtoNets](#attempt-1-post-hoc-attention-with-protonets)
    *   [Hypothesis and Method](#hypothesis-and-method)
    *   [Insights from Attempt 1](#insights-from-attempt-1)
    *   [Results (Loss, Accuracy, Explanations)](#results-loss-accuracy-explanations)
4.  [Inspiration: Learning from "This Looks Like That" (ProtoPNet)](#inspiration-learning-from-this-looks-like-that-protopnet)
    *   [ProtoPNet Architecture](#protopnet-architecture)
5.  [Our Hybrid Model: HEPN (Hybrid Explanable Prototypical Network)](#our-hybrid-model-hepn-hybrid-explanable-prototypical-network)
    *   [Core Idea and Hypothesis](#core-idea-and-hypothesis)
    *   [HEPN: Two Levels of Reasoning](#hepn-two-levels-of-reasoning)
    *   [Explaining via Generic Part Activations](#explaining-via-generic-part-activations)
    *   [Training & Overall Accuracy (HEPN)](#training--overall-accuracy-hepn)
    *   [HEPN Explanation Examples](#hepn-explanation-examples)
6.  [Lessons Learned](#lessons-learned)
7.  [Future Work](#future-work)
8.  [Summary of Exploration](#summary-of-exploration)

## Motivation: Why Explain AI Decisions?

Standard deep learning models are often treated as "black boxes," especially in complex tasks like fine-grained image recognition. This lack of interpretability poses several challenges:

*   **Trust & Reliability:** It's difficult to trust a prediction if we don't understand its basis, which is critical in domains like medicine or safety-critical systems.
*   **Debugging:** Understanding the model's reasoning is key to identifying and fixing errors.
*   **Scientific Discovery:** Interpretable models can reveal new insights about the data, such as discriminative features.

**The Goal of this Project:** To explore methods to make Few-Shot Image Recognition models more interpretable.

## The Challenge: Learning from Few Examples

*   **Few-Shot Learning (FSL):** The task of classifying new categories using only a small number (K-shot) of labeled examples.
*   **Prototypical Networks (ProtoNets):** A standard meta-learning approach for FSL.
    *   Learns a good embedding space.
    *   Calculates class prototypes as the mean of support set embeddings within each episode.
    *   Classifies query images based on their distance to these episode-specific prototypes.
*   **Limitation:** While effective, standard ProtoNets don't explain *why* a query image is considered close to a particular prototype (i.e., which features mattered for the decision).

## Attempt 1: Post-Hoc Attention with ProtoNets

### Hypothesis and Method

*   **Hypothesis:** Identify which parts of a query image are most similar to parts of the support images for the predicted class.
*   **Method:**
    1.  Use the convolutional features (before global pooling) from a trained ProtoNet encoder.
    2.  Calculate pairwise attention scores (e.g., scaled dot-product) between query image patches and support image patches.
    3.  Visualize the highest-scoring patch pairs.
*   **Implementation:** Utilized a ResNet encoder, with custom patch extraction and attention calculation logic.

### Insights from Attempt 1

*   **Pros:**
    *   Successfully highlights regions of high similarity between a query image and specific support examples.
    *   Provides spatial grounding for the similarity.
*   **Cons:**
    *   **Post-Hoc:** The explanation is calculated *after* classification and uses different features (convolutional maps) than the primary decision logic (global embeddings).
    *   Explains instance-to-instance similarity, not necessarily prototypicality for the class concept.
    *   Doesn't show what the model learned as a general "part" or concept representative of a class.

### Results (Loss, Accuracy, Explanations)

The model was trained, and its performance was tracked:

*   **Loss and Accuracy Plots:**

    ![image](https://github.com/user-attachments/assets/03a7be36-083e-4541-9e9d-37ffae47b93e)

*   **Explanation Example:**

    ![v2-3](https://github.com/user-attachments/assets/e5d667ec-f889-4b36-88e6-65f21c49acf2)
    *Example of attention-based explanation showing top-3 matching patches between a query image and support images.*

## Inspiration: Learning from "This Looks Like That" (ProtoPNet)

Inspired by the **ProtoPNet** paper (Chen et al., NeurIPS 2019), which proposes a model that reasons by comparing input image patches to learned, class-specific prototypical parts.

*   **Core Idea of ProtoPNet:**
    *   Learns global, class-specific prototypes as actual network parameters (learned image patches).
    *   Uses projection to make these prototypes directly visualizable (they are tied to specific training patches).
    *   The explanation is inherent to the classification process – similarity scores to these learned parts form the evidence.
*   **Effort:** Implemented the ProtoPNet architecture to understand its mechanics and potential for few-shot learning.
*   **Key Difference for FSL Context:** ProtoPNet is designed for standard supervised learning with fixed classes. In a meta-learning context for FSL, prototypes need to adapt to new, unseen classes with few examples. The original ProtoPNet uses class-specific prototypes, whereas FSL typically uses episode-specific prototypes.

### ProtoPNet Architecture

![Screenshot 2025-04-19 203457](https://github.com/user-attachments/assets/45175871-e8e8-4d96-ab27-ed5077534e0d)
*Diagram illustrating the architecture of a ProtoPNet model.*

## Our Hybrid Model: HEPN (Hybrid Explanable Prototypical Network)

### Core Idea and Hypothesis

To bridge the gap between the few-shot capabilities of ProtoNets and the inherent interpretability of ProtoPNet, a hybrid model, HEPN, was developed.

*   **Combine Strengths:**
    *   **Level 1:** Few-Shot capability of standard ProtoNets.
    *   **Level 2:** Interpretability from learned, generic parts (adapted for meta-learning).
*   **Hypothesis:** It's possible to learn generic, reusable part prototypes and use their activation patterns for explainable few-shot classification.

### HEPN: Two Levels of Reasoning

1.  **Encoder:** Outputs both global embeddings (for Level 1) and convolutional feature maps (for Level 2).
2.  **Level 1 (Global Reasoning):**
    *   Standard ProtoNet mechanism using global embeddings.
    *   Calculates episode-specific prototypes from support set global embeddings.
    *   Computes distance from query global embedding to these prototypes.
3.  **Level 2 (Novel Part Layer - Local Reasoning):**
    *   Contains learned, global, **generic part prototypes** (as `nn.Parameter`). These are not tied to specific classes initially but are learned to represent common visual patterns.
    *   Calculates **Part Activation Profiles (PAPs):** For a query image and for each support class (averaged over its support samples), it determines how strongly each generic part is activated. This is done by finding the max similarity between image patches and each generic part.
    *   Compares the query's PAP to each support class's average PAP using cosine similarity.
4.  **Combination:** The final decision combines the Level 1 (global distance) scores and Level 2 (part profile similarity) scores.
5.  **Part Learning:**
    *   **Projection:** Generic parts are projected onto patches from the meta-training set for visualization (to see what visual concept each generic part has learned).
    *   **Losses:**
        *   **Cluster Loss:** Encourages image patches to be close to at least one generic part prototype.
        *   **Diversity Loss:** Encourages the learned generic part prototypes to be distinct from each other.

### Explaining via Generic Part Activations

*   **Mechanism:** Grad-CAM is used to visualize *why* a specific generic part prototype was activated by the query image.
    *   The target score for Grad-CAM is the activation strength of a chosen generic part.
    *   This shows where in the image the features contributing to that part's activation are located.
*   **Note:** While this explanation method (Grad-CAM) is post-hoc (uses gradients after the forward pass), it is directly linked to the interpretable learned components (the generic parts) of the HEPN model.

### Training & Overall Accuracy (HEPN)

The HEPN model was trained, and various metrics were tracked:

![results](https://github.com/user-attachments/assets/915cb638-4987-40dd-857a-6bc2ff622fb6)

### HEPN Explanation Examples

*   **When It Works:**

    ![adv ml proj gradcam manual 1](https://github.com/user-attachments/assets/8306a38f-5323-40fc-a447-57ce8acbfaa2)
    *Example of HEPN explanation: Query image, its top activated generic parts, the class's prototype profile, and Grad-CAM visualizations for the most influential parts.*

*   **Challenges in Explainability:**

    ![download (4)](https://github.com/user-attachments/assets/28304217-41e6-4a42-acec-5f2d8a60e632)
    *Example illustrating challenges in HEPN explainability, where generic parts might not be perfectly aligned or discriminative for all fine-grained distinctions.*

## Lessons Learned

*   **Post-Hoc Attention (Attempt 1):**
    *   Simple and provides spatial grounding.
    *   Explains instance-to-instance similarity, not necessarily learned class concepts.
*   **ProtoPNet (Paper Inspiration):**
    *   Offers inherent explanation through class-specific parts.
    *   Designed for standard supervised learning, not directly for meta-learning's dynamic classes.
*   **HEPN (Our Hybrid Model):**
    *   A novel approach combining global (ProtoNet-style) and local (part-based) reasoning for few-shot learning.
    *   Uses generic learned parts suitable for a meta-learning context.
    *   Explanation is linked to these learned parts but relies on post-hoc Grad-CAM for visualization.
*   **Challenges with HEPN Explainability:**
    *   **Grad-CAM Stability/Noise:** Grad-CAM visualizations can sometimes be noisy or sensitive to minor input changes.
    *   **Discriminative Power of Generic Parts:** Generic parts might not always be sufficiently discriminative for very specific fine-grained classes compared to class-specific parts (as in original ProtoPNet). The "genericity" required for FSL can be a trade-off.
    *   **Balancing Complexity:** Balancing the influence of the two levels of reasoning (global vs. part-based) and tuning the regularization losses (cluster, diversity) is complex.

## Future Work

*   **Explore more robust CAM techniques:** Investigate alternatives like LayerCAM or Ablation CAM to potentially improve the stability and clarity of part activation visualizations.
*   **Refine part prototype losses:** Develop or refine loss functions to encourage the learning of more discriminative and diverse generic parts.
*   **Improve visualization techniques:** Beyond Grad-CAM, explore other methods to better understand and present how the learned parts contribute to the final decision in the few-shot context.

## Summary of Exploration

This project embarked on a journey to enhance the interpretability of few-shot learning models.
*   It began with exploring the need for interpretability and attempting post-hoc attention mechanisms on standard Prototypical Networks.
*   The principles of ProtoPNet, offering inherent explanations via learned parts, were studied and implemented as a source of inspiration.
*   This led to the development of **HEPN**, a novel hybrid approach that learns generic parts within a meta-learning framework, aiming to combine few-shot capability with part-based interpretability.
*   HEPN's performance and its (challenging but insightful) explainability using Grad-CAM were evaluated.

The emphasis of this project was a deep dive into the complexities of building explainable models, highlighting the trade-offs between different approaches (e.g., post-hoc vs. inherent, generic vs. specific parts) and the challenges in achieving consistently clear interpretations. The process yielded significant learning in the domain of interpretable few-shot learning.
