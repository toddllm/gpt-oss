#!/bin/bash

# Ollama Model Management Script
# Provides utilities for managing Ollama models

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_menu() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}       OLLAMA MODEL MANAGER${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
    echo "1. List installed models"
    echo "2. Pull a new model"
    echo "3. Remove a model"
    echo "4. Show model information"
    echo "5. Create custom model from Modelfile"
    echo "6. Export model"
    echo "7. Import model"
    echo "8. Update all models"
    echo "9. Show disk usage"
    echo "0. Exit"
    echo ""
    echo -n "Select an option: "
}

list_models() {
    echo -e "\n${GREEN}Installed Models:${NC}"
    ollama list
    echo ""
}

pull_model() {
    echo -n "Enter model name to pull (e.g., llama3.1:8b): "
    read model_name
    if [ -z "$model_name" ]; then
        echo -e "${RED}Model name cannot be empty${NC}"
        return
    fi
    echo -e "${YELLOW}Pulling $model_name...${NC}"
    ollama pull "$model_name"
    echo -e "${GREEN}✓ Model $model_name pulled successfully${NC}"
}

remove_model() {
    echo -e "\n${YELLOW}Available models:${NC}"
    ollama list | tail -n +2 | awk '{print "  • " $1}'
    echo ""
    echo -n "Enter model name to remove: "
    read model_name
    if [ -z "$model_name" ]; then
        echo -e "${RED}Model name cannot be empty${NC}"
        return
    fi
    echo -n "Are you sure you want to remove $model_name? (y/N): "
    read confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        ollama rm "$model_name"
        echo -e "${GREEN}✓ Model $model_name removed${NC}"
    else
        echo -e "${YELLOW}Removal cancelled${NC}"
    fi
}

show_model_info() {
    echo -e "\n${YELLOW}Available models:${NC}"
    ollama list | tail -n +2 | awk '{print "  • " $1}'
    echo ""
    echo -n "Enter model name to inspect: "
    read model_name
    if [ -z "$model_name" ]; then
        echo -e "${RED}Model name cannot be empty${NC}"
        return
    fi
    echo -e "\n${GREEN}Model Information for $model_name:${NC}"
    ollama show "$model_name"
}

create_custom_model() {
    echo -e "\n${YELLOW}Create Custom Model${NC}"
    echo "Enter path to Modelfile (or press Enter to use default Modelfile): "
    read modelfile_path
    
    if [ -z "$modelfile_path" ]; then
        modelfile_path="./Modelfile"
    fi
    
    if [ ! -f "$modelfile_path" ]; then
        echo -e "${RED}Modelfile not found at $modelfile_path${NC}"
        echo "Would you like to create a basic Modelfile? (y/N): "
        read create_new
        if [ "$create_new" = "y" ] || [ "$create_new" = "Y" ]; then
            echo -n "Enter base model (e.g., llama3.1:8b): "
            read base_model
            echo -n "Enter new model name: "
            read new_model_name
            
            cat > "$modelfile_path" << EOF
FROM $base_model

# Set custom parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

# Set a custom system prompt (optional)
SYSTEM "You are a helpful assistant."

# Custom template (optional)
TEMPLATE """{{ .System }}

User: {{ .Prompt }}