from itertools import groupby
from neo4j import GraphDatabase
from graph_src.graph_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


class DBInterviewer:
	def __init__(self, view_name="graphview"):
		self.EXTRA_FIELDS = {
			"drug": ["description", "indication"],
			"disease": [
				"mayo_causes", "mayo_complications", "mayo_prevention",
				"mayo_risk_factors", "mayo_symptoms", "mondo_definition"
			]
		}
		self.graphview = view_name
		self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
		self.Project()

	# Create GDS projection
	def Project(self):
		# Check if the graph projection already exists
		check_query = f"RETURN gds.graph.exists('{self.graphview}') AS exists"
		with self.driver.session() as session:
			result = session.run(check_query)
			exists = result.single()["exists"]
		if exists:
			return

		query = f"""
		CALL gds.graph.project(
			'{self.graphview}',
			'*',
			{{
				rels: {{
					type: '*',
					orientation: 'UNDIRECTED',
					aggregation: 'SINGLE'
				}}
			}}
		)
		"""
		with self.driver.session() as session:
			result = session.run(query)

	# Yen K-shortest paths
	def Yen(self, source, target, K):
		source_d = (str(source[1]), int(source[0]))
		target_d = (str(target[1]), int(target[0]))
		K = int(K)
		
		query = f"""
				MATCH (source:{source_d[0]} {{node_index: {source_d[1]}}})
				MATCH (target:{target_d[0]} {{node_index: {target_d[1]}}})
				CALL gds.shortestPath.yens.stream('{self.graphview}', {{
					sourceNode: source,
					targetNode: target,
					k: {K}
				}})
				YIELD index, path

				WITH index, path, nodes(path) AS ns
				UNWIND range(0, size(ns)-2) AS i
				WITH index, ns[i] AS fromNode, ns[i+1] AS toNode
				MATCH (fromNode)-[r]-(toNode)
				RETURN
					index AS pathNumber,
					fromNode,
					toNode,
					r.display_relation AS displayRelation
				"""
		
		with self.driver.session() as session:
			result = list(session.run(query))

		context = self.context_creator(result)

		return context

	def data_extractor(self, query_result):
		all_paths = [list(group) for _, group in groupby((record.values() for record in query_result), key=lambda r: r[0])]
		
		# all_paths = [
		# 	[
		# 		[0, A, B, "r1"],
		# 		[0, B, C, "r2"],
		# 		[0, C, D, "r3"]
		# 	],
		#	[
		#		[1, F, G, "r4"],
		# 		[1, G, H, "r5"]
		#	]
		# ]

		results = []

		for path in all_paths:
			# (A, B, r1), (B, C, r2), (C, D, r3) --> (A, r1), (B, r2), (C, r3), (D)
			for pair in path:
				num_path = pair[0]
				node = pair[1]
				display_rel = pair[3]
				
				node_type = list(node.labels)[0]
				node_name = node['node_name']
				
				data = {'num_path': num_path, 'display_relation': display_rel, 'node_type': node_type, 'node_name': node_name}

				fields = self.EXTRA_FIELDS.get(node_type, [])

				for key in fields:
					if key in node:
						data[key] = node[key]
				
				results.append(data)
			
			last_elem = path[-1]
			last_node = last_elem[2]
			last_node_type = list(last_node.labels)[0]
			last_node_name = last_node['node_name']
			last_datum = {'num_path': last_elem[0], 'node_type': last_node_type, 'node_name': last_node_name}
			
			fields = self.EXTRA_FIELDS.get(last_node_type, [])

			for key in fields:
				if key in last_node:
					last_datum[key] = last_node[key]

			results.append(last_datum)

		return results
	
	def data_formatter(self, all_paths):
		v_paths = {}  # dict {num_path: list of all verbalized pairs of nodes of a path}
		defs = {}  # dict {num_path: unique list of definitions} definitions appear only in the first path they are retrieved in
		global_defined_nodes = set()  # To keep track of definitions already printed before

		for i, curr_node in enumerate(all_paths):
			num_path = curr_node.get("num_path")

			if num_path not in v_paths:
				v_paths[num_path] = []
				defs[num_path] = []

			if "display_relation" in curr_node and i + 1 < len(all_paths):
				verbalization = self.verbalizer(curr_node, all_paths[i + 1])
				v_paths[num_path].append(verbalization)

			fields = self.EXTRA_FIELDS.get(curr_node["node_type"], [])

			if fields:
				# mondo_definition first
				if "mondo_definition" in fields:
					ordered_fields = ["mondo_definition"] + [f for f in fields if f != "mondo_definition"]
				else:
					ordered_fields = fields

				node_name = curr_node["node_name"]

				# Avoid redefining the same node in later paths
				if node_name not in global_defined_nodes:
					node_def = {}

					for field in ordered_fields:
						if field in curr_node and curr_node[field] and str(curr_node[field]).strip():
							label = field.replace("mayo_", "")
							label = label.replace("mondo_definition", "definition")

							node_def[label] = curr_node[field].strip()

					# If the node actually has attributes, store it
					if node_def:
						defs[num_path].append((node_name, node_def))
						global_defined_nodes.add(node_name)

		context = []

		for num_path in sorted(v_paths.keys()):
			# RELATIONS
			relations = "\n".join(v_paths[num_path])

			# DEFINITIONS
			defs_list = []
			if defs[num_path]:
				for node_name, node_def in defs[num_path]:
					defs_list.append((node_name, node_def))

			context.append((num_path, relations, defs_list))

		return context
	
	def verbalizer(self, node, next_node):
		return f'{node["node_name"]} has a {node["display_relation"]} relation with {next_node["node_name"]}'
	
	def context_creator(self, query_result):
		extracted_paths = self.data_extractor(query_result)
		formatted_data = self.data_formatter(extracted_paths)
		return formatted_data
