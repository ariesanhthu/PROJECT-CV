# Texture Analysis Summary

## Overall Statistics
- Samples analyzed: **3972**
- Patch-level texture samples: **7243**
- Classes represented in texture summary: **98**
- Mean foreground ratio per image: **0.5327**
- Mean foreground ratio per patch: **0.7037**

## Key Findings
- Smoothest class by score: **pudding** (`smoothness_mean=1.6068`)
- Most granular class by score: **salad** (`granularity_mean=3.9853`)
- Most crumbly-edge class by score: **cashew** (`crumbly_edge_mean=2.4213`)
- Hardest texture pair: **chicken duck vs fish** (`texture_similarity=0.9959`, `color_similarity=0.1768`, `cause_tag=texture-driven`)

## Top Classes By Smoothness
| class_name | smoothness_mean | n_patches | presence_count |
|---|---:|---:|---:|
| pudding | 1.6068 | 5 | 4 |
| dried cranberries | 1.5682 | 17 | 6 |
| coffee | 1.5651 | 98 | 99 |
| olives | 1.5265 | 50 | 28 |
| blueberry | 1.5159 | 120 | 131 |
| tea | 1.5155 | 19 | 17 |
| chocolate | 1.4783 | 85 | 37 |
| cherry | 1.4779 | 120 | 129 |
| watermelon | 1.4679 | 16 | 8 |
| candy | 1.4676 | 10 | 6 |

## Top Classes By Granularity
| class_name | granularity_mean | n_patches | presence_count |
|---|---:|---:|---:|
| salad | 3.9853 | 10 | 10 |
| lamb | 3.6610 | 63 | 57 |
| shellfish | 3.4690 | 26 | 11 |
| bean sprouts | 3.4304 | 16 | 14 |
| eggplant | 3.4208 | 12 | 10 |
| fig | 3.3276 | 13 | 5 |
| date | 3.3135 | 5 | 3 |
| pork | 3.2849 | 120 | 392 |
| rape | 3.2841 | 48 | 46 |
| fish | 3.2439 | 120 | 221 |

## Top Classes By Crumbly Edge
| class_name | crumbly_edge_mean | n_patches | presence_count |
|---|---:|---:|---:|
| cashew | 2.4213 | 6 | 2 |
| almond | 2.3286 | 38 | 18 |
| spring onion | 2.2961 | 120 | 96 |
| chocolate | 2.2843 | 85 | 37 |
| candy | 2.2108 | 10 | 6 |
| blueberry | 2.1775 | 120 | 131 |
| cilantro mint | 2.0492 | 120 | 504 |
| onion | 2.0448 | 120 | 299 |
| cherry | 2.0406 | 120 | 129 |
| walnut | 2.0274 | 20 | 11 |

## Top Hard Pairs
| pair | texture_similarity | color_similarity | cause_tag |
|---|---:|---:|---|
| chicken duck vs fish | 0.9959 | 0.1768 | texture-driven |
| fish vs pie | 0.9899 | 0.9798 | mixed |
| chicken duck vs pie | 0.9842 | 0.1854 | texture-driven |
| fish vs pork | 0.9789 | -0.4475 | texture-driven |
| chicken duck vs pork | 0.9781 | 0.6699 | texture-driven |
| onion vs spring onion | 0.9781 | -0.3445 | texture-driven |
| cake vs lamb | 0.9780 | 0.8060 | texture-driven |
| blueberry vs cherry | 0.9700 | 0.6751 | texture-driven |
| pie vs pork | 0.9690 | -0.4292 | texture-driven |
| fried meat vs pie | 0.9675 | -0.4166 | texture-driven |

## Actionables
- Classes flagged for `larger_crop`: **16**
- Classes flagged for `edge_detail_branch`: **53**
- Classes flagged for `group_sampling`: **40**
- Classes flagged for `annotation_audit`: **0**

## Generated Charts
- `outputs/charts/actionable_summary.png`
- `outputs/charts/leafy_greens_gallery.png`
- `outputs/charts/mushroom_related_gallery.png`
- `outputs/charts/pork_vs_steak_gallery.png`
- `outputs/charts/rice_vs_potato_gallery.png`
- `outputs/charts/texture_descriptives_top_classes.png`
- `outputs/charts/texture_embedding_pca_or_umap.png`
- `outputs/charts/texture_similarity_heatmap.png`
- `outputs/charts/top_hard_pair_gallery.png`
- `outputs/charts/top_hard_pairs_texture.png`
