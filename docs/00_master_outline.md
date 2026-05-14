# 视频生成领域教学文档集（2020–2026）总目录与写作计划

> 版本：v0.1（目录阶段）  
> 日期：2026-05-14  
> 目标：先定义完整可执行目录与章节顺序；下一阶段按目录逐章撰写（每篇关键论文与每个子问题文档均不少于 5000 字）。

---

## 0. 文档集总览

本教学文档集分为 6 个层级：

1. **导读与方法论层**：统一术语、阅读路径、评测规范、复现建议。  
2. **时间线与知识图谱层**：从 GAN→VAE/AR→Diffusion→Foundation/World Model 的技术演进。  
3. **关键论文精读层**：按“里程碑论文”逐篇展开，含架构、训练、实验、影响力。  
4. **子问题专题层**：围绕 motion、条件控制、长视频、效率、可控性等关键问题系统梳理。  
5. **工程与应用层**：训练实践、推理优化、部署、数据治理与安全。  
6. **总结与展望层**：趋势、开放问题、研究议程。

---

## 1. 顶层目录结构（文件树）

```text
video_generation_docs/
├─ 00_preface/
│  ├─ 00_readme.md
│  ├─ 01_how_to_use_this_series.md
│  └─ 02_notation_and_symbols.md
├─ 01_overview_and_timeline/
│  ├─ 01_evolution_2020_2026.md
│  ├─ 02_taxonomy_of_video_generation_methods.md
│  ├─ 03_benchmark_protocols_and_metrics.md
│  └─ 04_citation_and_influence_map.md
├─ 02_paper_deep_dive/
│  ├─ 00_reading_order.md
│  ├─ 01_gan_era/
│  │  ├─ 01_tgan.md
│  │  ├─ 02_mocogan.md
│  │  ├─ 03_stylegan_v_and_followups.md
│  │  └─ 04_limits_of_gan_video_generation.md
│  ├─ 02_autoregressive_and_tokenization/
│  │  ├─ 01_video_transformer.md
│  │  ├─ 02_videogpt.md
│  │  ├─ 03_maskgit_magvit_family.md
│  │  └─ 04_ar_vs_latent_token_models.md
│  ├─ 03_diffusion_foundations/
│  │  ├─ 01_vdm_and_latent_video_diffusion.md
│  │  ├─ 02_mcvd.md
│  │  ├─ 03_videodiffusion_text2video_zero.md
│  │  ├─ 04_videofusion.md
│  │  └─ 05_3d_unet_space_time_attention.md
│  ├─ 04_large_scale_t2v_systems/
│  │  ├─ 01_imagen_video.md
│  │  ├─ 02_phenaki_and_long_context.md
│  │  ├─ 03_emu_video.md
│  │  ├─ 04_movideo.md
│  │  ├─ 05_open_sora_like_pipelines.md
│  │  └─ 06_sora_veo_gen_series.md
│  ├─ 05_recent_surveys_2025_2026/
│  │  ├─ 01_text_to_video_generators_survey_2025.md
│  │  ├─ 02_video_diffusion_generation_review_2025.md
│  │  ├─ 03_survey_video_diffusion_models_2025.md
│  │  ├─ 04_controllable_video_generation_survey_2025.md
│  │  ├─ 05_evolution_of_video_generative_foundations_2026.md
│  │  └─ 06_efficient_video_diffusion_models_2026.md
│  └─ 06_appendix_paper_cards/
│     ├─ template_paper_card.md
│     └─ all_paper_cards_index.md
├─ 03_topic_monographs/
│  ├─ 01_motion_modeling.md
│  ├─ 02_text_conditioning_and_alignment.md
│  ├─ 03_multimodal_conditioning.md
│  ├─ 04_long_video_consistency_and_memory.md
│  ├─ 05_camera_control_and_scene_dynamics.md
│  ├─ 06_subject_identity_and_style_consistency.md
│  ├─ 07_controllable_generation_and_editing.md
│  ├─ 08_efficiency_distillation_and_acceleration.md
│  ├─ 09_data_engineering_and_captioning.md
│  ├─ 10_evaluation_and_benchmarking.md
│  └─ 11_safety_ethics_and_governance.md
├─ 04_system_design_and_practice/
│  ├─ 01_end_to_end_pipeline_design.md
│  ├─ 02_training_recipes.md
│  ├─ 03_inference_serving_and_cost.md
│  ├─ 04_productization_case_studies.md
│  └─ 05_failure_modes_and_debugging.md
├─ 05_global_assets/
│  ├─ 01_timeline_chart.md
│  ├─ 02_method_family_map.md
│  ├─ 03_term_index.md
│  ├─ 04_dataset_index.md
│  ├─ 05_metric_index.md
│  ├─ 06_reference_list.md
│  └─ 07_open_problems_and_future_directions.md
└─ 99_project_management/
   ├─ 01_writing_plan_and_milestones.md
   ├─ 02_quality_checklist.md
   └─ 03_update_log.md
```

---

## 2. 关键论文章节规划（按时间与技术脉络）

> 说明：每篇关键论文章节将统一包含以下固定小节（最终每章≥5000字）  
> A. 背景与问题定义  
> B. 模型架构与算法流程（公式/伪代码/ASCII图）  
> C. 训练策略与数据  
> D. 实验与Benchmark  
> E. 创新点、局限与失败案例  
> F. 与前后工作的因果关系  
> G. 衍生工作与产业应用  
> H. 影响力（引用量、被复现度、社区采用）

### 2.1 第一阶段（基础谱系，先修）

1. **MoCoGAN / TGAN 系列**（GAN 视频生成起点）  
2. **Video Transformer / VideoGPT**（离散token与自回归建模）  
3. **MCVD**（扩散在视频补全/生成中的关键节点）  
4. **Video Diffusion Models + Space-Time U-Net**（扩散范式成熟）

### 2.2 第二阶段（大模型化与系统化）

5. **Imagen Video**（级联扩散与高分辨率T2V）  
6. **VideoFusion**（跨帧一致性、扩散时空协同）  
7. **Emu Video**（多模态预训练迁移到视频）  
8. **MoVideo**（运动控制/主体与镜头可控机制）

### 2.3 第三阶段（Foundation / World Model 时代）

9. **Sora 系列**（长时空一致性与世界模拟趋势）  
10. **Veo 系列**（高质量文本到视频生产系统）  
11. **Gen 系列**（统一多模态生成接口/模型家族）

### 2.4 并行阅读（综述元分析）

12. 2025–2026 六篇综述（用户指定）作为“横向比较主线”：
   - Text‑to‑Video Generators: A Comprehensive Survey (2025)
   - Video Diffusion Generation: Comprehensive Review and Open Problems (2025)
   - Survey of Video Diffusion Models (2025)
   - Controllable Video Generation: A Survey (2025)
   - Evolution of Video Generative Foundations (2026)
   - Efficient Video Diffusion Models (2026)

---

## 3. 子问题专题文档规划（每篇≥5000字）

1. **Motion Modeling**：显式光流/隐式运动场/时空注意力/物理一致性  
2. **文本条件与语义对齐**：prompt 编码、对齐损失、指令跟随评估  
3. **多模态条件**：图像、音频、深度、分割、骨架、视频片段条件融合  
4. **长视频一致性**：记忆机制、分段生成、层次规划、故事线保持  
5. **可控镜头与场景动力学**：camera trajectory、3D先验、场景图控制  
6. **主体身份与风格一致性**：ID preserving、角色/服饰/风格稳定  
7. **可控生成与编辑**：局部重绘、时序编辑、轨迹控制、约束采样  
8. **效率优化**：蒸馏、步数压缩、缓存、token压缩、系统级加速  
9. **数据工程**：数据清洗、caption 生成、偏差与版权治理  
10. **评测体系**：自动指标、人工评测、任务导向评测与可重复性  
11. **安全伦理与治理**：深伪风险、溯源、水印、审核与策略

---

## 4. 总整合文档规划

1. **时间线图谱**：2020–2026 关键节点与技术断层  
2. **方法体系图**：GAN / AR / Diffusion / Hybrid / World Model 关系图  
3. **术语索引**：按“建模对象、网络结构、训练技巧、评测指标”分类  
4. **数据集索引**：规模、标注类型、许可、适用任务  
5. **指标索引**：FVD、IS、CLIPScore、Aesthetic、人评协议  
6. **参考文献总表**：BibTeX + 引用量 + 被引趋势  
7. **趋势总结与未来挑战**：长时一致性、物理规律、多智能体、交互式世界模型

---

## 5. 写作顺序（执行计划）

### Phase A：框架与元信息
- A1. 完成导读、术语、评测协议、时间线初版
- A2. 建立“论文卡片模板”（统一记录公式、架构图、实验表）

### Phase B：关键论文精读（主干）
- B1. GAN → AR → Diffusion 三阶段
- B2. Imagen Video / VideoFusion / Emu Video / MoVideo
- B3. Sora/Veo/Gen 与 open-source 复现路线

### Phase C：子问题专题（纵深）
- C1. motion + long video + controllability（优先）
- C2. multimodal + efficiency + evaluation

### Phase D：全局整合与校验
- D1. 引用量核验（Google Scholar / Semantic Scholar）
- D2. 交叉引用一致性检查（时间线与章节互链）
- D3. 术语索引、参考文献、未来挑战终稿

---

## 6. 每章统一模板（用于后续逐章生成）

```markdown
# [章节标题]

## 1. 历史定位与问题背景
## 2. 任务定义与符号系统
## 3. 核心方法
### 3.1 架构图
### 3.2 算法流程
### 3.3 关键公式与直觉解释
## 4. 训练细节与工程实现
## 5. 数据集与实验设置
## 6. 结果分析（定量+定性）
## 7. 与前后工作的联系
## 8. 局限、失败案例与改进方向
## 9. 衍生工作、应用与产业影响
## 10. 影响力与引用数据（含时间戳）
## 11. 小结与练习题
## 12. 参考文献
```

---

## 7. 质量标准（后续写作验收）

- 每篇论文章/专题章 **≥5000字**。  
- 每章至少包含：
  - 1 个时序演化图（ASCII/表格可替代）
  - 1 个方法流程图（ASCII/伪代码）
  - 2 张以上对比表（方法、数据集、指标）
- 所有“引用量/影响力”必须标注**统计时间**与**数据源**。  
- 所有 benchmark 数值需给出原始论文或权威复现实验来源。

---

## 8. 下一步（从目录进入正文）

按上述顺序，建议先从以下 3 篇开启正文：
1. `02_paper_deep_dive/02_autoregressive_and_tokenization/02_videogpt.md`  
2. `02_paper_deep_dive/03_diffusion_foundations/02_mcvd.md`  
3. `02_paper_deep_dive/04_large_scale_t2v_systems/01_imagen_video.md`

随后进入专题：
- `03_topic_monographs/01_motion_modeling.md`
- `03_topic_monographs/04_long_video_consistency_and_memory.md`

> 注：后续逐章写作阶段将联网核验引用量与benchmark原始出处，并在每章末附“数据更新时间”。
