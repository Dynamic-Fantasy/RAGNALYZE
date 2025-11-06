# RAGNALYZE Execution Flow
```
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#6366f1','primaryTextColor':'#fff','primaryBorderColor':'#4f46e5','lineColor':'#8b5cf6','secondaryColor':'#8b5cf6','tertiaryColor':'#ec4899','background':'#1e293b','mainBkg':'#3b82f6','secondBkg':'#8b5cf6','tertiaryBkg':'#ec4899','textColor':'#ffffff','nodeBorder':'#4f46e5','clusterBkg':'#334155','clusterBorder':'#6366f1','fontSize':'16px','nodeRadius':'15px'}}}%%

graph TB
    Start([Start Application]) --> APICheck{API Key<br/>Valid?}
    
    APICheck -->|No| APIDialog[API Key Dialog<br/>Validate & Save]
    APIDialog --> APICheck
    APICheck -->|Yes| UploadPDF[Upload PDF File]
    
    UploadPDF --> Configure[Configure Settings<br/>Model, Chunk Size, Retrieval K]
    Configure --> InitBtn[Click Initialize]
    
    InitBtn --> InitSystem[Initialize System]
    
    subgraph InitProcess ["<b style='font-size:20px'>INITIALIZATION PROCESS</b>"]
        InitSystem --> SavePDF[Save Temp PDF]
        SavePDF --> InitLLM[Initialize LLM<br/>ChatGroq Model]
        InitLLM --> LoadPages[Load PDF Pages<br/>PyPDFLoader]
        LoadPages --> SplitCheck{PDF Size<br/>Check}
        
        SplitCheck -->|Small ≤10 pages| SeqSplit[Sequential Split]
        SplitCheck -->|Large >10 pages| ParSplit[Parallel Split<br/>Multi-Processing]
        
        ParSplit -->|Success| Embeddings[Load Embeddings<br/>HuggingFace Model]
        ParSplit -->|Error| Fallback[Fallback to<br/>Sequential]
        Fallback --> Embeddings
        SeqSplit --> Embeddings
        
        Embeddings --> VectorDB[(Create VectorStore<br/>ChromaDB)]
        VectorDB --> CreateTools[Setup Tools<br/>Retriever, Show Feedbacks, Export]
        CreateTools --> BindTools[Bind Tools to LLM]
        BindTools --> BuildGraph[Build LangGraph<br/>Agent Workflow]
    end
    
    BuildGraph --> Ready[System Ready]
    
    Ready --> UserQuery[User Asks Question]
    
    subgraph AgentFlow ["<b style='font-size:20px'>AGENT PROCESSING FLOW</b>"]
        UserQuery --> LLMNode[LLM Node<br/>Process Query]
        LLMNode --> NeedTool{Tool Call<br/>Required?}
        
        NeedTool -->|No| DirectAnswer[Direct Answer]
        NeedTool -->|Yes| ToolExec[Execute Tool]
        
        ToolExec --> ToolType{Which<br/>Tool?}
        
        ToolType -->|Search| RetrieverTool[Retriever Tool<br/>Search PDF Content]
        ToolType -->|List| ShowTool[Show Feedbacks Tool<br/>List Multiple Items]
        ToolType -->|Export| ExportTool[Export Chat Tool<br/>Generate PDF]
        
        RetrieverTool --> VectorSearch[(Vector Search<br/>ChromaDB)]
        ShowTool --> VectorSearch
        
        VectorSearch --> ReturnResults[Return Results]
        ExportTool --> GeneratePDF[Generate PDF File]
        GeneratePDF --> ReturnResults
        
        ReturnResults --> LLMNode
        DirectAnswer --> DisplayAnswer[Display to User]
    end
    
    DisplayAnswer --> MoreQuestions{More<br/>Questions?}
    MoreQuestions -->|Yes| UserQuery
    MoreQuestions -->|No| ExportOption{Export<br/>Chat?}
    
    ExportOption -->|Yes| FinalExport[Export Conversation<br/>to PDF]
    ExportOption -->|No| End([End Session])
    FinalExport --> End
    
    style Start fill:#10b981,stroke:#059669,stroke-width:3px,color:#fff
    style End fill:#ef4444,stroke:#dc2626,stroke-width:3px,color:#fff
    style Ready fill:#10b981,stroke:#059669,stroke-width:3px,color:#fff
    style LLMNode fill:#6366f1,stroke:#4f46e5,stroke-width:3px,color:#fff
    style VectorDB fill:#8b5cf6,stroke:#7c3aed,stroke-width:3px,color:#fff
    style VectorSearch fill:#8b5cf6,stroke:#7c3aed,stroke-width:3px,color:#fff
    style ParSplit fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    style RetrieverTool fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff
    style ShowTool fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff
    style ExportTool fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff
    style DisplayAnswer fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    style InitProcess fill:#1e293b,stroke:#6366f1,stroke-width:3px
    style AgentFlow fill:#1e293b,stroke:#8b5cf6,stroke-width:3px
    ```
