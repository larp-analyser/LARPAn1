import networkx as nx
import networkx.algorithms.community as nx_comm
from datetime import datetime, timezone

UTC = timezone.utc

def build_networkx_context(username: str, user_graph: dict, group_graph: dict = None) -> str:
    """
    Constructs a NetworkX Directed Graph from user and group data,
    applies temporal decay to edge weights, calculates PageRank and community factions,
    and returns a structured mathematical dossier for the LLM.
    """
    G = nx.DiGraph()
    now = datetime.now(UTC)
    
    if not group_graph:
        group_graph = {"entities": [], "relationships": [], "last_updated": now.isoformat()}
        
    try: 
        user_age_days = (now - datetime.fromisoformat(user_graph.get("last_updated", now.isoformat()))).days
    except: 
        user_age_days = 0
        
    try: 
        group_age_days = (now - datetime.fromisoformat(group_graph.get("last_updated", now.isoformat()))).days
    except: 
        group_age_days = 0

    # Exponential decay formula: Half-life roughly ~6.5 days, floors at 10%
    user_decay = max(0.1, 0.9 ** user_age_days)
    group_decay = max(0.1, 0.9 ** group_age_days)
    
    for data, decay_factor in [(user_graph, user_decay), (group_graph, group_decay)]:
        for ent in data.get("entities", []):
            if isinstance(ent, str):
                node_id = ent
                new_attrs = "Unknown"
                node_type = "Unknown"
            else:
                node_id = ent.get("id")
                new_attrs = ent.get("attributes", "Unknown")
                node_type = ent.get("type", "Unknown")
            
            if not node_id:
                continue
                
            if node_id not in G:
                G.add_node(node_id, type=node_type, attributes=new_attrs)
            else:
                if new_attrs and new_attrs != "Unknown":
                    existing_attrs = G.nodes[node_id].get("attributes")
                    if not existing_attrs or existing_attrs == "Unknown":
                        G.nodes[node_id]["attributes"] = new_attrs
                    elif new_attrs not in str(existing_attrs):
                        G.nodes[node_id]["attributes"] += f" | {new_attrs}"
                
        for rel in data.get("relationships", []):
            src = rel.get("source")
            tgt = rel.get("target")
            rel_desc = rel.get("relation")
            base_weight = float(rel.get("intensity", 5.0))
            decayed_weight = base_weight * decay_factor
            
            if not src or not tgt:
                continue
                
            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] += decayed_weight
                if rel_desc not in G[src][tgt]['relation']:
                    G[src][tgt]['relation'] += f" | {rel_desc}"
            else:
                G.add_edge(src, tgt, relation=rel_desc, weight=decayed_weight)
            
    if username not in G:
        return f"--- TARGET DOSSIER: {username} ---\nNo known network connections. Target is socially isolated."

    try:
        social_scores = nx.pagerank(G, weight='weight', max_iter=500, tol=1e-6)
        target_score = social_scores.get(username, 0.0)
        ranked_users = sorted(social_scores.items(), key=lambda x: x[1], reverse=True)
        rank_index = next((i for i, v in enumerate(ranked_users) if v[0] == username), len(ranked_users))
        social_status = f"Rank {rank_index + 1} out of {len(ranked_users)} active entities."
    except Exception as e:
        target_score, social_status = 0.0, "Unknown"

    try:
        undirected_G = G.to_undirected()
        # CPU Guardian Check: Bypass O(n^2 log n) community detection on micro-VM if graph is too large
        if len(undirected_G.nodes) <= 2000:
            factions = list(nx_comm.greedy_modularity_communities(undirected_G))
            user_faction = next((list(f) for f in factions if username in f), [])
            faction_str = ", ".join([u for u in user_faction if u != username]) if len(user_faction) > 1 else "Lone Wolf"
        else:
            faction_str = "Unknown (Graph too large for deep analysis)"
    except:
        faction_str = "Unknown"
        
    context_lines = []
    node_attrs = G.nodes[username].get("attributes", "Unknown")
    
    context_lines.append(f"--- TARGET DOSSIER: {username} ---")
    context_lines.append(f"CORE TRAITS: {node_attrs}")
    context_lines.append(f"SOCIAL RANK (PageRank): {target_score:.4f} ({social_status})")
    context_lines.append(f"DETECTED FACTION / ALLIES: {faction_str}")
    
    edges_dict = { (u, v): d for u, v, d in G.in_edges(username, data=True) }
    edges_dict.update({ (u, v): d for u, v, d in G.out_edges(username, data=True) })
    edges = [ (u, v, d) for (u, v), d in edges_dict.items() ]
    
    if edges:
        context_lines.append("\nACTIVE RELATIONSHIPS (Weighted by Time/Decay):")
        edges.sort(key=lambda x: x[2].get('weight', 0), reverse=True)
        for source, target, data in edges[:5]:
            w = data.get('weight', 0)
            status = "[FADING]" if w < 2.0 else "[ACTIVE]"
            context_lines.append(f"- {status} {source} [{data['relation']}] {target} (Relevance: {w:.1f})")
            
    return "\n".join(context_lines)
