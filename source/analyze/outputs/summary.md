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

## Notes
- Chart preview/export is now handled by the last cell in `source/analyze/texture.ipynb`.
- When that cell is run, all `.png` files in `outputs/charts` will be displayed in-notebook and a fresh markdown summary can be regenerated automatically.
Build sample metadata: 100%|██████████| 3972/3972 [05:43<00:00, 11.56it/s]
Extract texture patches: 100%|██████████| 3972/3972 [04:05<00:00, 16.15it/s]
Handcrafted texture: 100%|██████████| 7243/7243 [04:12<00:00, 28.70it/s]
Done. Saved to: outputs\texture_full_colab
Samples: 3972 | Patches: 7243 | Features: 7243

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>split</th>
      <th>stem</th>
      <th>class_id</th>
      <th>class_name</th>
      <th>image_path</th>
      <th>mask_path</th>
      <th>patch_id</th>
      <th>center_x</th>
      <th>center_y</th>
      <th>patch_size</th>
      <th>...</th>
      <th>b_std</th>
      <th>h_mean</th>
      <th>h_std</th>
      <th>s_mean</th>
      <th>s_std</th>
      <th>v_mean</th>
      <th>v_std</th>
      <th>smoothness_score</th>
      <th>granularity_score</th>
      <th>crumbly_edge_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train</td>
      <td>00000000</td>
      <td>21</td>
      <td>chicken duck</td>
      <td>..\dataset\foodseg103-full\train\img\00000000.jpg</td>
      <td>..\dataset\foodseg103-full\train\mask\00000000...</td>
      <td>train_00000000_c21_00</td>
      <td>117</td>
      <td>282</td>
      <td>96</td>
      <td>...</td>
      <td>5.667098</td>
      <td>12.989149</td>
      <td>2.494067</td>
      <td>243.157776</td>
      <td>13.397788</td>
      <td>126.619141</td>
      <td>20.771776</td>
      <td>1.390442</td>
      <td>2.950347</td>
      <td>1.238498</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train</td>
      <td>00000000</td>
      <td>80</td>
      <td>rice</td>
      <td>..\dataset\foodseg103-full\train\img\00000000.jpg</td>
      <td>..\dataset\foodseg103-full\train\mask\00000000...</td>
      <td>train_00000000_c80_00</td>
      <td>332</td>
      <td>183</td>
      <td>96</td>
      <td>...</td>
      <td>18.341358</td>
      <td>20.935764</td>
      <td>1.250433</td>
      <td>93.608505</td>
      <td>20.393522</td>
      <td>187.003143</td>
      <td>8.397468</td>
      <td>1.489226</td>
      <td>3.138168</td>
      <td>1.239475</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train</td>
      <td>00000000</td>
      <td>88</td>
      <td>snow peas</td>
      <td>..\dataset\foodseg103-full\train\img\00000000.jpg</td>
      <td>..\dataset\foodseg103-full\train\mask\00000000...</td>
      <td>train_00000000_c88_00</td>
      <td>77</td>
      <td>90</td>
      <td>96</td>
      <td>...</td>
      <td>20.810081</td>
      <td>40.532551</td>
      <td>4.338658</td>
      <td>196.204041</td>
      <td>38.997917</td>
      <td>94.336250</td>
      <td>18.072552</td>
      <td>1.322345</td>
      <td>3.229113</td>
      <td>1.298943</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train</td>
      <td>00000001</td>
      <td>27</td>
      <td>cucumber</td>
      <td>..\dataset\foodseg103-full\train\img\00000001.jpg</td>
      <td>..\dataset\foodseg103-full\train\mask\00000001...</td>
      <td>train_00000001_c27_00</td>
      <td>337</td>
      <td>129</td>
      <td>96</td>
      <td>...</td>
      <td>36.286697</td>
      <td>22.354971</td>
      <td>7.096622</td>
      <td>142.733597</td>
      <td>42.394238</td>
      <td>159.583725</td>
      <td>34.992580</td>
      <td>1.002656</td>
      <td>3.455008</td>
      <td>1.271784</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train</td>
      <td>00000001</td>
      <td>60</td>
      <td>onion</td>
      <td>..\dataset\foodseg103-full\train\img\00000001.jpg</td>
      <td>..\dataset\foodseg103-full\train\mask\00000001...</td>
      <td>train_00000001_c60_00</td>
      <td>224</td>
      <td>329</td>
      <td>96</td>
      <td>...</td>
      <td>27.745111</td>
      <td>14.435101</td>
      <td>29.131008</td>
      <td>166.186203</td>
      <td>44.500580</td>
      <td>102.812485</td>
      <td>22.866341</td>
      <td>1.486943</td>
      <td>2.605527</td>
      <td>1.717467</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>