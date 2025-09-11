# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum

DB_CASSANDRA_CONSISTENCY_LEVEL = "db.cassandra.consistency_level"
"""
The consistency level of the query. Based on consistency values from [CQL](https://docs.datastax.com/en/cassandra-oss/3.0/cassandra/dml/dmlConfigConsistency.html).
"""

DB_CASSANDRA_COORDINATOR_DC = "db.cassandra.coordinator.dc"
"""
The data center of the coordinating node for a query.
"""

DB_CASSANDRA_COORDINATOR_ID = "db.cassandra.coordinator.id"
"""
The ID of the coordinating node for a query.
"""

DB_CASSANDRA_IDEMPOTENCE = "db.cassandra.idempotence"
"""
Whether or not the query is idempotent.
"""

DB_CASSANDRA_PAGE_SIZE = "db.cassandra.page_size"
"""
The fetch size used for paging, i.e. how many rows will be returned at once.
"""

DB_CASSANDRA_SPECULATIVE_EXECUTION_COUNT = "db.cassandra.speculative_execution_count"
"""
The number of times a query was speculatively executed. Not set or `0` if the query was not executed speculatively.
"""

DB_CASSANDRA_TABLE = "db.cassandra.table"
"""
The name of the primary Cassandra table that the operation is acting upon, including the keyspace name (if applicable).
Note: This mirrors the db.sql.table attribute but references cassandra rather than sql. It is not recommended to attempt any client-side parsing of `db.statement` just to get this property, but it should be set if it is provided by the library being instrumented. If the operation is acting upon an anonymous table, or more than one table, this value MUST NOT be set.
"""

DB_CONNECTION_STRING = "db.connection_string"
"""
Deprecated: "Replaced by `server.address` and `server.port`.".
"""

DB_COSMOSDB_CLIENT_ID = "db.cosmosdb.client_id"
"""
Unique Cosmos client instance id.
"""

DB_COSMOSDB_CONNECTION_MODE = "db.cosmosdb.connection_mode"
"""
Cosmos client connection mode.
"""

DB_COSMOSDB_CONTAINER = "db.cosmosdb.container"
"""
Cosmos DB container name.
"""

DB_COSMOSDB_OPERATION_TYPE = "db.cosmosdb.operation_type"
"""
CosmosDB Operation Type.
"""

DB_COSMOSDB_REQUEST_CHARGE = "db.cosmosdb.request_charge"
"""
RU consumed for that operation.
"""

DB_COSMOSDB_REQUEST_CONTENT_LENGTH = "db.cosmosdb.request_content_length"
"""
Request payload size in bytes.
"""

DB_COSMOSDB_STATUS_CODE = "db.cosmosdb.status_code"
"""
Cosmos DB status code.
"""

DB_COSMOSDB_SUB_STATUS_CODE = "db.cosmosdb.sub_status_code"
"""
Cosmos DB sub status code.
"""

DB_ELASTICSEARCH_CLUSTER_NAME = "db.elasticsearch.cluster.name"
"""
Represents the identifier of an Elasticsearch cluster.
"""

DB_ELASTICSEARCH_NODE_NAME = "db.elasticsearch.node.name"
"""
Deprecated: Replaced by `db.instance.id`.
"""

DB_ELASTICSEARCH_PATH_PARTS_TEMPLATE = "db.elasticsearch.path_parts"
"""
A dynamic value in the url path.
Note: Many Elasticsearch url paths allow dynamic values. These SHOULD be recorded in span attributes in the format `db.elasticsearch.path_parts.<key>`, where `<key>` is the url path part name. The implementation SHOULD reference the [elasticsearch schema](https://raw.githubusercontent.com/elastic/elasticsearch-specification/main/output/schema/schema.json) in order to map the path part values to their names.
"""

DB_INSTANCE_ID = "db.instance.id"
"""
An identifier (address, unique name, or any other identifier) of the database instance that is executing queries or mutations on the current connection. This is useful in cases where the database is running in a clustered environment and the instrumentation is able to record the node executing the query. The client may obtain this value in databases like MySQL using queries like `select @@hostname`.
"""

DB_JDBC_DRIVER_CLASSNAME = "db.jdbc.driver_classname"
"""
Deprecated: Removed as not used.
"""

DB_MONGODB_COLLECTION = "db.mongodb.collection"
"""
The MongoDB collection being accessed within the database stated in `db.name`.
"""

DB_MSSQL_INSTANCE_NAME = "db.mssql.instance_name"
"""
The Microsoft SQL Server [instance name](https://docs.microsoft.com/sql/connect/jdbc/building-the-connection-url?view=sql-server-ver15) connecting to. This name is used to determine the port of a named instance.
Note: If setting a `db.mssql.instance_name`, `server.port` is no longer required (but still recommended if non-standard).
"""

DB_NAME = "db.name"
"""
This attribute is used to report the name of the database being accessed. For commands that switch the database, this should be set to the target database (even if the command fails).
Note: In some SQL databases, the database name to be used is called "schema name". In case there are multiple layers that could be considered for database name (e.g. Oracle instance name and schema name), the database name to be used is the more specific layer (e.g. Oracle schema name).
"""

DB_OPERATION = "db.operation"
"""
The name of the operation being executed, e.g. the [MongoDB command name](https://docs.mongodb.com/manual/reference/command/#database-operations) such as `findAndModify`, or the SQL keyword.
Note: When setting this to an SQL keyword, it is not recommended to attempt any client-side parsing of `db.statement` just to get this property, but it should be set if the operation name is provided by the library being instrumented. If the SQL statement has an ambiguous operation, or performs more than one operation, this value may be omitted.
"""

DB_REDIS_DATABASE_INDEX = "db.redis.database_index"
"""
The index of the database being accessed as used in the [`SELECT` command](https://redis.io/commands/select), provided as an integer. To be used instead of the generic `db.name` attribute.
"""

DB_SQL_TABLE = "db.sql.table"
"""
The name of the primary table that the operation is acting upon, including the database name (if applicable).
Note: It is not recommended to attempt any client-side parsing of `db.statement` just to get this property, but it should be set if it is provided by the library being instrumented. If the operation is acting upon an anonymous table, or more than one table, this value MUST NOT be set.
"""

DB_STATEMENT = "db.statement"
"""
The database statement being executed.
"""

DB_SYSTEM = "db.system"
"""
An identifier for the database management system (DBMS) product being used. See below for a list of well-known identifiers.
"""

DB_USER = "db.user"
"""
Username for accessing the database.
"""


class DbCassandraConsistencyLevelValues(Enum):
    ALL = "all"
    """all."""
    EACH_QUORUM = "each_quorum"
    """each_quorum."""
    QUORUM = "quorum"
    """quorum."""
    LOCAL_QUORUM = "local_quorum"
    """local_quorum."""
    ONE = "one"
    """one."""
    TWO = "two"
    """two."""
    THREE = "three"
    """three."""
    LOCAL_ONE = "local_one"
    """local_one."""
    ANY = "any"
    """any."""
    SERIAL = "serial"
    """serial."""
    LOCAL_SERIAL = "local_serial"
    """local_serial."""


class DbCosmosdbConnectionModeValues(Enum):
    GATEWAY = "gateway"
    """Gateway (HTTP) connections mode."""
    DIRECT = "direct"
    """Direct connection."""


class DbCosmosdbOperationTypeValues(Enum):
    INVALID = "Invalid"
    """invalid."""
    CREATE = "Create"
    """create."""
    PATCH = "Patch"
    """patch."""
    READ = "Read"
    """read."""
    READ_FEED = "ReadFeed"
    """read_feed."""
    DELETE = "Delete"
    """delete."""
    REPLACE = "Replace"
    """replace."""
    EXECUTE = "Execute"
    """execute."""
    QUERY = "Query"
    """query."""
    HEAD = "Head"
    """head."""
    HEAD_FEED = "HeadFeed"
    """head_feed."""
    UPSERT = "Upsert"
    """upsert."""
    BATCH = "Batch"
    """batch."""
    QUERY_PLAN = "QueryPlan"
    """query_plan."""
    EXECUTE_JAVASCRIPT = "ExecuteJavaScript"
    """execute_javascript."""


class DbSystemValues(Enum):
    OTHER_SQL = "other_sql"
    """Some other SQL database. Fallback only. See notes."""
    MSSQL = "mssql"
    """Microsoft SQL Server."""
    MSSQLCOMPACT = "mssqlcompact"
    """Microsoft SQL Server Compact."""
    MYSQL = "mysql"
    """MySQL."""
    ORACLE = "oracle"
    """Oracle Database."""
    DB2 = "db2"
    """IBM Db2."""
    POSTGRESQL = "postgresql"
    """PostgreSQL."""
    REDSHIFT = "redshift"
    """Amazon Redshift."""
    HIVE = "hive"
    """Apache Hive."""
    CLOUDSCAPE = "cloudscape"
    """Cloudscape."""
    HSQLDB = "hsqldb"
    """HyperSQL DataBase."""
    PROGRESS = "progress"
    """Progress Database."""
    MAXDB = "maxdb"
    """SAP MaxDB."""
    HANADB = "hanadb"
    """SAP HANA."""
    INGRES = "ingres"
    """Ingres."""
    FIRSTSQL = "firstsql"
    """FirstSQL."""
    EDB = "edb"
    """EnterpriseDB."""
    CACHE = "cache"
    """InterSystems Caché."""
    ADABAS = "adabas"
    """Adabas (Adaptable Database System)."""
    FIREBIRD = "firebird"
    """Firebird."""
    DERBY = "derby"
    """Apache Derby."""
    FILEMAKER = "filemaker"
    """FileMaker."""
    INFORMIX = "informix"
    """Informix."""
    INSTANTDB = "instantdb"
    """InstantDB."""
    INTERBASE = "interbase"
    """InterBase."""
    MARIADB = "mariadb"
    """MariaDB."""
    NETEZZA = "netezza"
    """Netezza."""
    PERVASIVE = "pervasive"
    """Pervasive PSQL."""
    POINTBASE = "pointbase"
    """PointBase."""
    SQLITE = "sqlite"
    """SQLite."""
    SYBASE = "sybase"
    """Sybase."""
    TERADATA = "teradata"
    """Teradata."""
    VERTICA = "vertica"
    """Vertica."""
    H2 = "h2"
    """H2."""
    COLDFUSION = "coldfusion"
    """ColdFusion IMQ."""
    CASSANDRA = "cassandra"
    """Apache Cassandra."""
    HBASE = "hbase"
    """Apache HBase."""
    MONGODB = "mongodb"
    """MongoDB."""
    REDIS = "redis"
    """Redis."""
    COUCHBASE = "couchbase"
    """Couchbase."""
    COUCHDB = "couchdb"
    """CouchDB."""
    COSMOSDB = "cosmosdb"
    """Microsoft Azure Cosmos DB."""
    DYNAMODB = "dynamodb"
    """Amazon DynamoDB."""
    NEO4J = "neo4j"
    """Neo4j."""
    GEODE = "geode"
    """Apache Geode."""
    ELASTICSEARCH = "elasticsearch"
    """Elasticsearch."""
    MEMCACHED = "memcached"
    """Memcached."""
    COCKROACHDB = "cockroachdb"
    """CockroachDB."""
    OPENSEARCH = "opensearch"
    """OpenSearch."""
    CLICKHOUSE = "clickhouse"
    """ClickHouse."""
    SPANNER = "spanner"
    """Cloud Spanner."""
    TRINO = "trino"
    """Trino."""
