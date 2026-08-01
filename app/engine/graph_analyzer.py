import networkx as nx
import networkx.algorithms.community as nx_comm
from datetime import datetime, timezone

UTC = timezone.utc

# Global RAM Cache for the actual mathematical graph object
_digraph_cache = {}

def build_networkx_context(username: str, user_graph: dict, group_graph: dict = None) -> str:
    """
    Constructs a NetworkX Directed Graph from user and group data,
    applies temporal decay to edge weights, calculates PageRank and community factions,
    and returns a structured mathematical dossier for the LLM.
    """
    now = datetime.now(UTC)
    
    if not group_graph:
        group_graph = {"entities": [], "relationships": [], "last_updated": now.isoformat()}
        
    # Generate a unique cache signature
    user_update_time = user_graph.get("last_updated") or "INITIALIZED"
    group_update_time = group_graph.get("last_updated") or "INITIALIZED"
    cache_key = f"{username}_{user_update_time}_{group_update_time}"
    
    if cache_key in _digraph_cache:
        G = _digraph_cache[cache_key]
    else:
        G = nx.DiGraph()
        
        for data in [user_graph, group_graph]:
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
                
                # Calculate decay specifically for this isolated edge using its last_seen timestamp
                edge_last_seen = rel.get("last_seen") or data.get("last_updated") or now.isoformat()
                try:
                    edge_age_days = (now - datetime.fromisoformat(edge_last_seen)).days
                except:
                    edge_age_days = 0
                    
                edge_decay_factor = max(0.1, 0.9 ** edge_age_days)
                decayed_weight = base_weight * edge_decay_factor
                
                if not src or not tgt:
                    continue
                    
                if G.has_edge(src, tgt):
                    G[src][tgt]['weight'] += decayed_weight
                    if rel_desc not in G[src][tgt]['relation']:
                        G[src][tgt]['relation'] += f" | {rel_desc}"
                else:
                    G.add_edge(src, tgt, relation=rel_desc, weight=decayed_weight)
                    
        # Save to RAM
        _digraph_cache[cache_key] = G
        
        # Prevent RAM leak: clear old signatures from cache for this user
        keys_to_delete = [k for k in _digraph_cache if k.startswith(f"{username}_") and k != cache_key]
        for k in keys_to_delete:
            del _digraph_cache[k]

    if username not in G:
        return f"--- TARGET DOSSIER: {username} ---\nNo known network connections. Target is socially isolated."

    try:
        social_scores = nx.pagerank(G, weight='weight')
        target_score = social_scores.get(username, 0.0)
        ranked_users = sorted(social_scores.items(), key=lambda x: x[1], reverse=True)
        rank_index = next((i for i, v in enumerate(ranked_users) if v[0] == username), len(ranked_users))
        social_status = f"Rank {rank_index + 1} out of {len(ranked_users)} active entities."
    except Exception as e:
        target_score, social_status = 0.0, "Unknown"

    try:
        undirected_G = G.to_undirected()
        factions = list(nx_comm.greedy_modularity_communities(undirected_G))
        user_faction = next((list(f) for f in factions if username in f), [])
        faction_str = ", ".join([u for u in user_faction if u != username]) if len(user_faction) > 1 else "Lone Wolf"
    except:
        faction_str = "Unknown"
        
    context_lines = []
    node_attrs = G.nodes[username].get("attributes", "Unknown")
    
    context_lines.append(f"--- TARGET DOSSIER: {username} ---")
    context_lines.append(f"CORE TRAITS: {node_attrs}")
    context_lines.append(f"SOCIAL RANK (PageRank): {target_score:.4f} ({social_status})")
    context_lines.append(f"DETECTED FACTION / ALLIES: {faction_str}")
    
    # Extract 2-Hop Subgraph using NetworkX ego_graph
    # radius=2 extracts the user + direct neighbors + neighbors of neighbors
    if username in G:
        ego_G = nx.ego_graph(G, username, radius=2, center=True)
        edges = [(u, v, d) for u, v, d in ego_G.edges(data=True)]
    else:
        edges = []
    
    if edges:
        context_lines.append("\nEXPANDED NETWORK RELATIONSHIPS (2-Hop Depth, Decay-Weighted):")
        # Sort edges by time-decayed weight
        edges.sort(key=lambda x: x[2].get('weight', 0), reverse=True)
        
        # INCREASED CAP: Top 25 most relevant relationships (up from 10)
        for source, target, data in edges[:25]:
            w = data.get('weight', 0)
            status = "[FADING]" if w < 2.0 else "[ACTIVE]"
            
            # Tag whether it is a direct (1-hop) or indirect (2-hop) connection
            hop_tag = "[DIRECT]" if (source == username or target == username) else "[INDIRECT]"
            
            context_lines.append(
                f"- {status} {hop_tag} {source} [{data.get('relation', 'connected')}] {target} (Relevance: {w:.1f})"
            )
            
    return "\n".join(context_lines)
