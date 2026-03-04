**INSTRUCTIONS**
<br><br>

Execute:
```bash
conda create -n GraphMedRag python=3.11.13
conda activate GraphMedRag
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
```

<br><br>
Download the files: *nodes.csv, edges.csv, disease_features.csv, drug_features.csv* from https://zitniklab.hms.harvard.edu/projects/PrimeKG/

Put them into the *graph_src/data* folder and execute the *data_preparation.ipynb* notebook to fix them

<br><br>
Then:
1. Install neo4j (utilized version: **2025.12.1**)
2. Download neo4j gds and apoc libraries and put them into */var/lib/neo4j/plugins* folder (utilized versions: **apoc-5.26.12-core**, **neo4j-graph-data-science-2.22.0**)
3. Put all the fixed csv files into the */var/lib/neo4j/import* folder
4. Modify */etc/neo4j/neo4j.conf*:
   
   Uncomment
   ```bash
     server.directories.import=/var/lib/neo4j/import
     dbms.security.auth_enabled=false
     server.default_listen_address=0.0.0.0
     server.default_advertised_address=localhost
     server.bolt.enabled=true
     server.bolt.listen_address=:7687
     server.http.enabled=true
     server.http.listen_address=:7474
   ```
   Add
   ```bash
     dbms.security.procedures.unrestricted=apoc.*
     dbms.security.procedures.allowlist=apoc.*,gds.*
   ```
7. Export:
   ```bash
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_DATABASE=neo4j
   NEO4J_PASSWORD=yourpassword
   ```
8. Start Neo4j database:
   ```bash
   sudo neo4j start
   ```
 
10. Execute this queries:
    ```cypher
    // Indexes creation
    CREATE CONSTRAINT gene_protein_index_unique
    FOR (n:gene_protein)
    REQUIRE n.node_index IS UNIQUE;
    
    CREATE CONSTRAINT drug_index_unique
    FOR (n:drug)
    REQUIRE n.node_index IS UNIQUE;
    
    CREATE CONSTRAINT disease_protein_index_unique
    FOR (n:disease)
    REQUIRE n.node_index IS UNIQUE;
    
    CREATE CONSTRAINT biological_process_index_unique
    FOR (n:biological_process)
    REQUIRE n.node_index IS UNIQUE;
    
    CREATE CONSTRAINT molecular_function_index_unique
    FOR (n:molecular_function)
    REQUIRE n.node_index IS UNIQUE;
    
    CREATE CONSTRAINT cellular_component_index_unique
    FOR (n:cellular_component)
    REQUIRE n.node_index IS UNIQUE;
    
    CREATE CONSTRAINT exposure_protein_index_unique
    FOR (n:exposure)
    REQUIRE n.node_index IS UNIQUE;
    
    CREATE CONSTRAINT pathway_index_unique
    FOR (n:pathway)
    REQUIRE n.node_index IS UNIQUE;
    
    CREATE CONSTRAINT anatomy_protein_index_unique
    FOR (n:anatomy)
    REQUIRE n.node_index IS UNIQUE;
    
    CREATE CONSTRAINT effect_phenotype_index_unique
    FOR (n:effect_phenotype)
    REQUIRE n.node_index IS UNIQUE;
    ```
   
    ```cypher
    // Nodes upload
    LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row FIELDTERMINATOR ','
    CALL apoc.create.node([row.node_type], {
        node_index: toInteger(row.node_index),
        node_id: row.node_id,
        node_name: row.node_name,
        node_source: row.node_source
    }) YIELD node
    RETURN count(node);
    ```
    
    ```cypher
    // Drugs properties upload:
    LOAD CSV WITH HEADERS FROM 'file:///drug_features_fixed.csv' AS row FIELDTERMINATOR ','
    WITH row, toInteger(row.node_index) AS idx
    MATCH (d:drug {node_index: idx})
    CALL apoc.create.setProperties(
      d,
      [
        'description','half_life','indication','mechanism_of_action','protein_binding',
        'pharmacodynamics','state','atc_1','atc_2','atc_3','atc_4',
        'category','group','pathway','molecular_weight','tpsa','clogp'
      ],
      [
        row.description,row.half_life,row.indication,row.mechanism_of_action,row.protein_binding,
        row.pharmacodynamics,row.state,row.atc_1,row.atc_2,row.atc_3,row.atc_4,
        row.category,row.group,row.pathway,row.molecular_weight,row.tpsa,row.clogp
      ]
    ) YIELD node
    RETURN count(node);
    ```

    ```cypher
    // Diseases properties upload:
    LOAD CSV WITH HEADERS FROM 'file:///disease_features_fixed.csv' AS row FIELDTERMINATOR ','
    WITH row, toInteger(row.node_index) AS idx
    MATCH (d:disease {node_index: idx})
    CALL apoc.create.setProperties(
      d,
      [
        'mondo_id','mondo_name','group_id_bert','group_name_bert','mondo_definition',
        'umls_description','orphanet_definition','orphanet_prevalence','orphanet_epidemiology',
        'orphanet_clinical_description','orphanet_management_and_treatment','mayo_symptoms',
        'mayo_causes','mayo_risk_factors','mayo_complications','mayo_prevention','mayo_see_doc'
      ],
      [
        row.mondo_id,row.mondo_name,row.group_id_bert,row.group_name_bert,row.mondo_definition,
        row.umls_description,row.orphanet_definition,row.orphanet_prevalence,row.orphanet_epidemiology,
        row.orphanet_clinical_description,row.orphanet_management_and_treatment,row.mayo_symptoms,
        row.mayo_causes,row.mayo_risk_factors,row.mayo_complications,row.mayo_prevention,row.mayo_see_doc
      ]
    ) YIELD node
    RETURN count(node);
    ```

      ```cypher
      // Edges upload
      CALL apoc.periodic.iterate(
         "
         LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
         RETURN 
         toInteger(row.x_index) AS xIndex, 
         toInteger(row.y_index) AS yIndex, 
         row.relation AS type, 
         row.display_relation AS display
         ",
         "
         MATCH (a:anatomy|biological_process|cellular_component|disease|drug|effect_phenotype|exposure|gene_protein|molecular_function|pathway {node_index: xIndex})
         MATCH (b:anatomy|biological_process|cellular_component|disease|drug|effect_phenotype|exposure|gene_protein|molecular_function|pathway {node_index: yIndex})
         
         CALL apoc.merge.relationship(a, type, {}, {display_relation:display}, b) YIELD rel
         
         RETURN rel
         ",
         {batchSize:100000,parallel:false}
      );
      ```

<br><br>
Example of execution:

Run all the tests for each dataset using concurrently COT, RAG and GRAG
```bash
python3 predictor.py
```
