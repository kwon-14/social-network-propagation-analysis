import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ==============================
# 配置环境：解决中文乱码、画布设置
# ==============================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(12, 9))

# ==============================
# 1. 数据读取与标准化预处理（DS核心流程）
# ==============================
df = pd.read_csv("sample_data.csv")

# 数据清洗：去空、去重，保证网络拓扑干净
df = df.dropna()
df = df.drop_duplicates(subset=["source", "target"])

print("===== 数据预处理完成 =====")
print(f"有效互动边数量：{len(df)}")
print(df.head())

# ==============================
# 2. 构建加权无向社交网络图
# ==============================
G = nx.Graph()

# 逐条加载权重边
for _, row in df.iterrows():
    G.add_edge(
        row["source"],
        row["target"],
        weight=row["weight"]
    )

# ==============================
# 3. 三大核心网络指标计算（保留重点指标）
# ==============================
# 1. 度中心性：局部活跃度
degree_central = nx.degree_centrality(G)

# 2. 介数中心性：传播桥梁能力
between_central = nx.betweenness_centrality(G, weight="weight")

# 3. PageRank：全局影响力权重排序
pagerank = nx.pagerank(G, alpha=0.85)

# ==============================
# 4. 结果结构化输出，生成分析表格
# ==============================
analysis_result = pd.DataFrame({
    "user_id": list(G.nodes()),
    "degree_centrality": [degree_central[i] for i in G.nodes()],
    "betweenness_centrality": [between_central[i] for i in G.nodes()],
    "pagerank_score": [pagerank[i] for i in G.nodes()]
})

# 按全局影响力排序
sorted_result = analysis_result.sort_values("pagerank_score", ascending=False)

print("\n===== 全局影响力用户TOP3 =====")
print(sorted_result.head(3))

# ==============================
# 5. 可视化社交传播网络
# ==============================
pos = nx.spring_layout(G, seed=2024)
node_size = [v * 3500 for v in degree_central.values()]

nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.85)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=12)

plt.title("Social Network Propagation Structure | 社交传播网络拓扑分析", fontsize=15)
plt.axis("off")
plt.tight_layout()
plt.savefig("screenshots/network_result.png", dpi=300)
plt.show()