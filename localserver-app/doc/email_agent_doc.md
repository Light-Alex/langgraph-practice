# 1. 在 Studio UI 里测试 email_agent

## 1.1. 流程

1. 打开 Studio & 选 graph

   - 地址：<https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024>
   - 左上角 graph 下拉里选 email_agent（别选成模板的 agent）。
   - 中间会显示 7 个节点的流程图。

2. 在 Input 面板填入邮件（JSON）

   Studio 的输入面板就是个 JSON 编辑器，填 EmailAgentState 的输入字段。先测场景 A（不会暂停，最简单）：

   ```json
   {
     "email_content": "Hi, how do I reset my password? I forgot it.",
     "sender_email": "user@example.com",
     "email_id": "E001"
   }
   ```

3. 点 Run

   点播放按钮（建议用 streaming 模式），能看到节点逐个点亮：read_email → classify_intent → search_documentation → draft_response → send_reply。点任意节点可看它的输入/输出，右侧 State 面板能看到 classification、draft_response 等。

4. 测 interrupt（这个 agent 的核心看点）

   把输入换成紧急邮件：

   ```json
   {
     "email_content": "I was charged TWICE for my subscription! Critical, refund me immediately!",
     "sender_email": "angry@example.com",
     "email_id": "E002"
   }
   ```

   跑到 human_review 时图会暂停，Studio 弹出中断提示和 Resume 对话框。这就是 email_agent.py:178 那个 interrupt() 抛出的内容。你在 Resume 框里填它等待的决策（注意是 interrupt() 返回的那个 dict 的结构）：

   ```json
   {
     "approved": true,
     "edited_response": "已核实，退款已发起，3-5 个工作日到账。"
   }
   ```

   点 Resume，图继续走 send_reply → END，draft_response 就变成你填的这段。填 approved: false 则直接到 END（模拟人工接管）。

## 1.2. 两点提醒

- 路由非确定性：分类是 glm-4-flash 实时判断的，每次跑结果可能略有不同（我 REST 测时 B 被分成了 bug/critical 而非 billing，但都进了 human_review）。想稳定复现某条路径，靠邮件措辞引导分类。
- 热重载：dev server 监听文件变化（日志里能看到 "changes detected"），你改 email_agent.py 存盘后它会自动重载，不用重启。
