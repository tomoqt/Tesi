# Tesi

---

## 1. Latent Test-time Compute

**Overview:**
Recent literature has demonstrated that it is possible to pretrain a model so that certain layers exhibit inner recurrence—i.e., you can sample the number of loops for which a layer is executed during training, and then adjust that layer’s recurrence at test time to monotonically increase performance.

**Two possible branches:**

* **Dynamics & Early Exit Methods**

  * Study the latent space of these recurrent loops as a dynamical system.
  * Characterize early‐exit strategies (when and how to stop looping) in order to optimize inference time vs. accuracy tradeoffs.
* **Weight‐sharing & Compression**

  * Analyze these models from the perspective of weight compression—how reusing the same parameters in loops effectively compresses the network.
  * Examine memory savings and performance impacts when the same layer parameters are reused multiple times.

**Repository:**
[github.com/tomoqt/Latent\_ttc](https://github.com/tomoqt/Latent_ttc)

---

## 2. Learned Mixed‐Curvature Models

**Overview:**
Transformer activations tend to be anisotropic. Prior work has shown hyperbolic activations can mitigate anisotropy by embedding representations in spaces of constant negative curvature. In this project, you will project activations into a constant‐curvature manifold (hyperbolic, spherical, or Euclidean), perform the standard Transformer operations in the tangent space, and then map back to the manifold.

**Key components:**

* **Per‐Layer / Per‐Head Curvature Learning**

  * Each layer (or even each attention head) learns its own curvature parameter during training.
  * Investigate how learned curvature evolves across layers and how it impacts representational geometry and downstream performance.
* **Implementation Details**

  * Efficient projection to/from manifold (log and exp maps).
  * Numerical stability considerations (e.g., avoiding vanishing gradients near curvature boundaries).

**Repository:**
[github.com/tomoqt/hyperbolic\_transformer](https://github.com/tomoqt/hyperbolic_transformer)

---

## 3. Hyperattention

**Overview:**
In standard Transformers, depth is the only mechanism by which representations of token **tuples** (higher‐order relationships) are formed. However, stacking many layers is computationally expensive and limits parallelism. This project proposes “lifting” the attention graph to a higher‐order simplicial complex so that multi‐token interactions can be modeled in fewer steps.

**Key challenges:**

* **Gating Mechanism (Mixture of Topologies)**

  * Design a gating function that routes a subset of tokens (or token‐pairs/tuples) into higher‐order processing while keeping memory usage bounded.
  * Explore how to dynamically select which simplices (e.g., edges, triangles, tetrahedra) to activate per input.
* **Efficient Tensor Operations**

  * Use tensor‐decomposition techniques (e.g., CP, Tucker) to perform multilinear attention updates in higher‐order spaces without incurring exponential memory blow‐up.
  * Benchmark trade‐offs between expressivity and computational overhead.
  * 
**Repository**
[github.com/tomoqt/hyperattention](https://github.com/tomoqt/hyperattention)

---

## 4. Entropix‐Steered GRPO

**Overview:**
Modern post‐training reinforcement learning techniques—such as DAPO (Data‐Augmented Policy Optimization) and GRPO (Generalized Replay‐based Policy Optimization)—often use relative advantages between rollouts as the reward signal. In this project, you will explore leveraging entropy (or even higher‐order surprisal moments) of the policy’s output distribution to better navigate the exploration vs. exploitation tradeoff, especially in “group generation”.



*(No repository yet; this can be implemented using existing RL frameworks such as Verifiers or custom code.)*

---

## 5. Oversmoothing vs. Oversquashing in Transformers

**Overview:**
Transformers can be viewed as Graph Attention Networks (GATs) that diffuse a vector‐valued function $V$ over a fully‐connected graph weighted by attention scores. This project investigates the fundamental tradeoff between oversmoothing (where node representations become indistinguishable) and oversquashing (where long‐range dependencies are “squashed” due to limited capacity), focusing particularly on:

* **The role of softmax** in regulating graph connectivity (i.e., how attention scores affect spectral gap).
* **Spectral gap analysis** of the attention‐weighted graph for different softmax temperature settings.
* Propose techniques (e.g., adaptive temperature, sparsity‐inducing priors) to mitigate both oversmoothing and oversquashing simultaneously.

**Reading:**

* Original paper on oversquashing: arXiv:2212.02374
  [arxiv.org/abs/2212.02374](https://arxiv.org/abs/2212.02374)

*(You can build experiments based on existing Transformer codebases and graph analysis libraries like NetworkX or PyTorch Geometric.)*

---

## 6. GRPO for Molecular Structure Elucidation

**Overview:**
We want to leverage a tokenized representation of NMR spectra (e.g., discretized peaks encoded as “tokens”) and then post‐train a language‐model‐style architecture (e.g., a Transformer) to predict the corresponding SMILES or SELFIES  string of the molecule directly—treating chemical structure as a token generation problem. We will use GRPO (Generalized Replay‐based Policy Optimization) to fine-tune in an RL framework, optimizing for both structural validity and spectral fidelity.

**Steps:**

1. **Data Preprocessing**

   * Tokenize NMR spectra into discrete bins (e.g., “peak @ δ 7.26, J=8.5 Hz” → token).
   * Pair with ground‐truth SMILES/SELFIES for each spectrum.
2. **Model Architecture**

   * Base Transformer (e.g., pretrained on chemical corpora or from scratch).
   * Policy outputs probability over SMILES vocabulary.
3. **RL Fine‐Tuning with GRPO**

   * Define reward that combines:

     * Validity (does generated SMILES → valid molecule?).
     * Spectral reconstruction error (simulate NMR from predicted SMILES and compare).
   * Use GRPO to optimize generation policy under replayed rollouts of predicted SMILES sequences.
4. **Evaluation**

   * Evaluate top‐k accuracy (does the true molecule appear among top k predictions?).
   * Structural similarity metrics (e.g., Tanimoto on fingerprints).
   * Downstream task: feeding predicted structure to property‐prediction models.

**Repository:**
[github.com/tomoqt/multimodal](https://github.com/tomoqt/multimodal.git)

---
## 7. Turing Machine Design from Natural Language: Benchmark
**Overview:**
This project introduces a benchmark to evaluate the algorithmic reasoning capabilities of large language models (LLMs) by tasking them with designing functional Turing machines from formal problem statements. The benchmark is built on a unique dataset of algorithmic problems from a long-running Italian national programming competition

**What’s already built:**
* Corpus Extraction Pipeline: A multimodal parsing script processes scanned and typeset competition PDFs into structured JSONL format, capturing problem statements, examples, and metadata.
* Data: ~300 problems are already parsed, covering 20+ editions of the national competition, each labeled by edition and difficulty level.
* Test Case Generator: A test case synthesis suite uses the Gemini API to generate diverse and challenging test cases that go beyond the examples given in the original problems.
* Turing Machine Simulator: The benchmark relies on the official JSTMSimulator, a JavaScript-based Turing machine simulator used in the original competition, faithfully reproducing the intended semantics and evaluation criteria. Serves as the ground truth execution environment. Supports full machine configuration: states, transitions, multi-symbol tapes, head movements, and halting logic. A dedicated Python wrapper (simulator.py) provides a programmatic bridge to the JS engine, enabling automated evaluation within the benchmark loop.
* Evaluation Framework: An orchestrator coordinates the loop of: prompting an LLM to generate a Turing machine candidate; simulating it using the custom interpreter (JSTMSimulator); collecting execution traces and iteratively refining the machine based on failures.

**Research Questions**
The central goal of this benchmark is to evaluate whether large language models (LLMs) can perform robust algorithmic reasoning by synthesizing fully functional Turing machines from natural language specifications.

**Work in Progress**
While the dataset and simulator are already available, the benchmark framework is being completed with the following evaluation modes: Zero-shot, Few-shot, Chain-of-Thought , Tool-augmented.

**Contact**
* federico.califano@uniroma1.it
* federico.califano@aisparks.it

---
## 8. REBUS: A truly multi-modal language-vision cloze test.

**Overview:** You might be familiar with
[REBUS puzzles](https://en.wikipedia.org/wiki/Rebus), specifically those made by
[La Settimana Enigmistica](https://www.lasettimanaenigmistica.com/) (SE). These
can be viewed as a form of multimodal (text and image)
[cloze](https://en.wikipedia.org/wiki/Cloze_test) test, a test whose unimodal
text-format variant is inspiration for the MLM pretraining objective central to
the
[distributional hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics)
in NLP which spurred the progress in LLMs in the last decade.

REBUS puzzles from SE are very difficult for humans, and current VLMs do
terrible on them. They require a range of capabilities such as localization,
multimodal understanding and reasoning.

In general there is demand for _hard_ benchmarks, so building a REBUS-style
benchmark would be a welcome and fun addition to the field.

**Goals**

1. Build the benchmark
   - we probably cannot scrape SE, for copyright reasons
   - however, we don't need to, we live in >2025, we can just generate
     REBUS-style puzzles using generative image models (e.g. diffusion, see
     [flux](https://github.com/black-forest-labs/flux) and the associated
     ecosystem around these models, like [ComfyUI](https://www.comfy.org/))
   - 50 to 200 samples is enough for a benchmark
2. Evaluate existing VLMs on the benchmark
3. (Optional) Try hill-climbing the benchmark
   - Would require a training set, which we can generate using the generative
     image models mentioned above, going beyond the 200 samples of evaluation.
   - we can try various post-training strategies on the VLMs, e.g.
     [GRPO](https://x.com/__sunil_kumar_/status/1900365571016806793)
4. (Optional) Perform experiments on the effects of tokenization on the
   performance on the benchmark
   - Might require a family of models trained on the same data but with
     different tokenization strategies, similar to what was done with
     [pythia](https://arxiv.org/abs/2304.01373). This could kind of be a project
     of its own (just training a family of models with different tokenization)
5. (Optional) What if we viewed this more as a ~ pre/midtraining task specifically
   made for VLMs? Risky but possibly interesting.

**Related Work**

- https://scale.com/leaderboard/enigma_eval
- https://cavendishlabs.org/rebus/
- ZeroBench
- TextGames: https://arxiv.org/abs/2502.18431
- Enigmata https://x.com/siyu_yuan_/status/1927248765712285776

**Possible models to evaluate**

- molmo
- Nvlm
- O1
- Gpt4o
- Claude 3.5 sonnet
- moondream

**Repository:**
[github.com/thesofakillers/rebus](https://github.com/thesofakillers/rebus)

---
### How to Choose

* **Theoretically inclined?**

  * **Oversmoothing vs. Oversquashing** or **Latent Test‐time Compute** (dynamics, spectral theory).
* **Geometry & Manifold methods?**

  * **Learned Mixed‐Curvature Models** (hyperbolic projections, curvature learning).
* **Higher-order graph structures?**

  * **Hyperattention** (simplicial complexes, tensor decompositions).
* **Reinforcement Learning focus?**

  * **Entropix-Steered GRPO** (entropy‐based reward design) or **GRPO for Molecular Structure Elucidation** (multimodal RL & chemistry).

Feel free to adapt any project to your background—e.g., you could combine **Hyperattention** ideas with **Mixed‐Curvature Models** for a truly geometric higher­-order Transformer.
