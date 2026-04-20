<claude-mem-context>
# Memory Context

# [discount-tire-recommendation-system] recent context, 2026-04-20 2:43am MST

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 45 obs (16,370t read) | 576,605t work | 97% savings

### Apr 19, 2026
26 11:54p 🔵 discount-tire-recommendation-system project file structure fully mapped
27 " 🔵 9 bugs found in GNN model architecture and training loop review
28 11:58p 🔵 Full code review confirmed: actual codebase is clean — no bugs found after deep re-read
29 " 🔵 Inductive GNN cold-start inference implemented in scripts/inference.py
### Apr 20, 2026
35 12:06a ⚖️ 6-step plan to eliminate train/val/test data leakage in GNN training pipeline
36 12:15a 🔵 BPRSampler and trainer evaluation wiring fully mapped across all source files
37 12:16a 🔴 Data leakage fix steps 1 & 2: ReviewEdgeSplit dataclass and train-only graph view added to sampler.py
39 " 🔴 Train-only graph view propagated through trainer.py and scripts/evaluate.py
41 " ✅ Smoke test updated to verify train graph edge count is less than full graph
43 12:24a 🔵 sampler.py structure: ReviewEdgeSplit (part 1) and BPRSampler (part 2) fully mapped
45 12:25a 🔵 Trainer.py architecture: three-loss training loop with leakage-free graph
46 " 🔵 Smoke test confirmed passing — split sizes, train graph, and all three losses verified
47 12:27a 🔴 BPR negative sampling now excludes all train-split reviews, not just positives
48 " 🔴 scripts/evaluate.py now uses sampler.train_data for all model encoding calls
52 12:37a 🟣 tests/test_split_integrity.py added — regression suite for train/val/test leakage prevention
53 12:38a 🟣 test_split_integrity.py passes — all 4 leakage regression checks confirmed green
55 1:00a 🔄 HGTEncoder user nodes switched from per-node Embedding to shared type_seed parameter
56 " 🟣 HGTLayer gains edge-attribute conditioning on attention logits and message gates
57 " 🔴 inference.py cold-start hack removed — type_seed makes manual embedding expansion obsolete
58 " 🔴 HGTEncoder edge_attr_dict comprehension fixed to use data.edge_types instead of data.edge_items()
62 1:01a ✅ HGTLayer edge conditioning weights switched from zero-init to Xavier uniform init
63 " 🟣 test_split_integrity.py gains type_seed, cold-start, and edge_attr sensitivity regression tests
66 1:02a 🔵 cold-start test fails — checked_encode identity assertion conflicts with recommend_new_user's deep copy
68 " 🔵 Training smoke test passes cleanly after type_seed and edge_attr conditioning architecture changes
69 " 🔴 test_split_integrity.py monkeypatch teardown added to fix cold-start test scope
70 1:05a ✅ Split integrity test suite passes after monkeypatch teardown fix — all regression checks green
73 " ✅ README.md updated to reflect train/val/test leakage fix and type_seed architecture
76 " ✅ Docstring and comment updates across hgt_layer.py, trainer.py, and inference.py for accuracy
77 1:08a 🔵 scripts/evaluate.py passes user_positives_list (list[list]) to std_evaluate — pre-existing type mismatch confirmed still present
79 1:10a 🔄 scripts/evaluate.py refactored — imports hoisted, BPRSampler default args, init_metric_sums helper added
80 " 🔵 evaluation.py accepts list[list[int]] despite list[set[int]] type annotation — iteration works for both
81 1:13a 🔴 scripts/inference.py fixed to run inference on train-only graph instead of full graph
83 " 🔴 scripts/evaluate.py CLI expanded with model arch, split-seed, and flexible --ks flags
86 " ✅ scripts/evaluate.py CLI fully parameterized — model arch, split seed, rating threshold, and flexible K cutoffs
88 1:14a ✅ Final regression test pass — all split integrity checks green after full session refactor
90 1:21a ✅ Training loop configured to 200 steps/epoch for 2 epochs
92 1:35a 🔵 Training device confirmed as CPU-only on development machine
95 2:07a 🔵 User raised concern: validation prediction catalog may exclude training items
97 2:16a 🔵 Validation/test targets contain substantial cold-start tire nodes not seen in training
98 2:18a 🔴 ReviewEdgeSplit now guarantees at least one train review edge per tire
100 " 🟣 Split integrity test now asserts all val/test target tires appear in training graph
102 " 🔵 Post-fix verification: all 5864 tires now covered in training, zero cold-start val/test targets
104 2:19a 🔵 Split integrity regression tests pass after per-tire reservation fix
105 2:26a ⚖️ Cluster refresh cadence under consideration — every 2 epochs
106 2:36a 🟣 tqdm progress bar added to scripts/train.py training loop

Access 577k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>