<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>医疗问答智能助手项目简介</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: #f5f7f9;
            color: #333;
            line-height: 1.6;
            padding: 20px;
            max-width: 1000px;
            margin: 0 auto;
        }
        
        header {
            background: linear-gradient(135deg, #4c8bf5, #1e5cb3);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        h1 {
            font-size: 2.2rem;
            margin-bottom: 1rem;
        }
        
        .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        section {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        h2 {
            color: #4c8bf5;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f0f0f0;
        }
        
        h3 {
            color: #555;
            margin: 1.2rem 0 0.7rem 0;
        }
        
        ul {
            list-style-type: none;
            padding-left: 1rem;
        }
        
        li {
            margin-bottom: 0.5rem;
            position: relative;
            padding-left: 1.5rem;
        }
        
        li:before {
            content: "•";
            color: #4c8bf5;
            font-weight: bold;
            position: absolute;
            left: 0;
        }
        
        .tech-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }
        
        .tech-card {
            background: #f8f9fa;
            padding: 1.2rem;
            border-radius: 8px;
            border-left: 4px solid #4c8bf5;
        }
        
        .tech-card h3 {
            color: #4c8bf5;
            margin-top: 0;
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.2rem;
            margin: 1.2rem 0;
        }
        
        .feature-card {
            background: #f0f7ff;
            padding: 1.2rem;
            border-radius: 8px;
            border: 1px solid #d0e3ff;
        }
        
        .feature-card h3 {
            color: #1e5cb3;
            margin-top: 0;
        }
        
        .value-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.2rem;
            margin: 1.2rem 0;
        }
        
        .value-card {
            background: #e6f7ff;
            padding: 1.2rem;
            border-radius: 8px;
            border: 1px solid #bae7ff;
        }
        
        .value-card h3 {
            color: #0066cc;
            margin-top: 0;
        }
        
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            color: #777;
            font-size: 0.9rem;
            border-top: 1px solid #eee;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            
            h1 {
                font-size: 1.8rem;
            }
            
            .tech-grid, .feature-grid, .value-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>医疗问答智能助手项目简介</h1>
        <p class="subtitle">基于Neo4j知识图谱和T5语言模型的智能问答系统</p>
    </header>

    <section>
        <h2>项目概述</h2>
        <p>医疗问答智能助手是一个基于Neo4j知识图谱和T5语言模型的智能问答系统，旨在解决大语言模型在医疗领域的"幻觉"问题，为用户提供准确可靠的医疗信息服务。</p>
        <p>该系统由东北大学秦皇岛分校管理学院信息管理与信息系统专业学生团队开发，采用"知识图谱+生成模型"的创新架构，有效结合了结构化知识的准确性和生成模型的灵活性。</p>
    </section>

    <section>
        <h2>核心技术</h2>
        <div class="tech-grid">
            <div class="tech-card">
                <h3>知识图谱技术</h3>
                <p>使用Neo4j图数据库构建结构化医疗知识网络，存储疾病、症状、药品等实体及其关系。</p>
            </div>
            <div class="tech-card">
                <h3>自然语言处理</h3>
                <p>采用T5模型进行自然语言生成，经过医疗领域专业数据微调，提供准确的问题回答。</p>
            </div>
            <div class="tech-card">
                <h3>混合智能架构</h3>
                <p>结合知识图谱检索和语言模型生成的混合方案，优先使用知识图谱，模型作为补充。</p>
            </div>
        </div>
    </section>

    <section>
        <h2>主要功能</h2>
        <div class="feature-grid">
            <div class="feature-card">
                <h3>用户管理</h3>
                <p>支持用户注册、登录和管理功能，保障系统安全性和个性化体验。</p>
            </div>
            <div class="feature-card">
                <h3>知识图谱检索</h3>
                <p>基于Neo4j的医疗实体关系查询与可视化，提供直观的知识探索界面。</p>
            </div>
            <div class="feature-card">
                <h3>智能问答</h3>
                <p>优先使用知识图谱提供精准答案，图谱未覆盖时调用T5生成回答，确保问题解决率。</p>
            </div>
        </div>
    </section>

    <section>
        <h2>技术特点</h2>
        <ul>
            <li>采用"知识图谱优先，模型兜底"的策略，确保回答准确性</li>
            <li>支持疾病症状查询、药品推荐、治疗方案咨询等多种医疗问答场景</li>
            <li>提供直观的知识图谱可视化界面，增强用户体验</li>
            <li>结合规则匹配和深度学习，提高问题理解准确率</li>
            <li>模块化设计，便于扩展和维护</li>
        </ul>
    </section>

    <section>
        <h2>应用价值</h2>
        <div class="value-grid">
            <div class="value-card">
                <h3>辅助医疗决策</h3>
                <p>可辅助医生快速检索医疗知识，提升诊疗效率和准确性。</p>
            </div>
            <div class="value-card">
                <h3>患者教育</h3>
                <p>为患者提供可靠的医疗常识解答，缓解医疗资源紧张问题。</p>
            </div>
            <div class="value-card">
                <h3>技术示范</h3>
                <p>为AI在专业领域的可靠应用提供了实践范例，具有推广价值。</p>
            </div>
        </div>
    </section>

    <div class="footer">
        <p>东北大学秦皇岛分校管理学院 · 信息管理与信息系统专业</p>
        <p>© 2025 医疗问答智能助手项目团队</p>
    </div>
</body>
</html>
