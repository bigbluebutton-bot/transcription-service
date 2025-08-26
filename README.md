# --- Corrected Startup Script ---

# First, ensure the variables from your .env file are loaded into your shell
# You can do this by running `export $(grep -v '^#' .env | xargs)` or similar
# before executing the rest of this script.

# Make the directory for the keyfile
echo "Creating mongo keyfile directory..."
mkdir -p ./data/mongo

# Generate the keyfile
echo "Generating keyfile..."
openssl rand -base64 756 > ./data/mongo/keyfile

# Set read-only permissions for the owner
chmod 400 ./data/mongo/keyfile

# Change the owner to the mongodb user's ID (999)
# This is a key step that requires sudo
echo "Setting keyfile ownership..."
sudo chown 999:999 ./data/mongo/keyfile

# STEP 1: Bring up ONLY the database services first
echo "Starting MongoDB and Redis nodes..."
docker compose up -d

# Wait for the containers to initialize
echo "Waiting for services to initialize (20s)..."
sleep 20

# STEP 2: Initiate the MongoDB replica set
echo "Initiating MongoDB replica set..."
docker compose exec mongo1 mongosh -u "admin" -p "admin" --authenticationDatabase admin --eval 'rs.initiate({_id: "rs0", members: [{_id: 0, host: "172.30.40.131:27017"}, {_id: 1, host: "172.30.40.131:27018"}, {_id: 2, host: "172.30.40.131:27019"}]})'

# Create Redis login data
mkdir -p ./data/redis/acl
echo "user admin on >admin allcommands allkeys" > ./data/redis/acl/users.acl
chmod -R 777 ./data/redisinsight


# STEP 3: Create the Redis cluster
echo "Creating Redis cluster..."
docker compose exec redis-1 redis-cli --cluster create 172.30.40.131:6379 172.30.40.131:6382 172.30.40.131:6383 172.30.40.131:6384 172.30.40.131:6385 172.30.40.131:6386 --cluster-replicas 1 --cluster-yes

# Done
echo "---"
echo "Setup complete! Services are running."
echo "Mongo Express: http://localhost:8081"
echo "Redis Insight: http://localhost:5540"

# STEP 5: Setup venv
uv venv
source .venv/bin/activate
uv pip install -e .

# STEP 6: Run the application
python main.py

