run_model.ipynb
- first part of codebase (https://github.com/makotu1208/open-problems-multimodal-3rd-solution/tree/main): model dictionary, prediction loop
- get model output on public data (=codebase) and on private data


preprocessing_sampling.ipynb
1. Get sample data with 212 columns (128 svd + 84 handselected genes)
- get 5/50/n samples per cell type: get_samples()
- save resulting df: X_test_shap_....h5ad
- prepare private test data: svd, also apply get_samples(), save private_test_input_sample.h5ad
2. Same steps for sample data with 148 columns (64 svd + 84 handselected genes)


shap.ipynb
- load model #16
- explainer = shap.KernelExplainer(model, medianX_train))
- get shap_values = explainer.shap_values(X_test_shap)
- save shap_values_16_n_samples.npy\
=> shap values for 212 features! 128 svd components + 84 handselected genes
- same steps for model #17


svd_backpropagation.ipynb
- get feature attributions for all genes instead of svd components
- load in normalized svd components svd_comp_norm
- get_attr_all_features() creates attr_all_22085_genes


plotting.ipynb
- plot shap_values for models #16 and #17 on 50 samples oer cell type; (public and) private data
- plot attr_all_22085_genes for models #16 and #17, use same samples as above -> backpropagate svd contributions
- get_plot_per_cell_type() plots attributions coloured by cell type


attribution_analysis.ipynb
- gene attribution distributions
- get genes with higher mean attributions
- plot distributions


140_classes.ipynb:
- handle and analyse all 140 protein classes instead of just 1
