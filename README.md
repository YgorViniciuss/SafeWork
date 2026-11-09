# SafeWork

Monitoramento inteligente de EPIs em zonas de risco com OpenCV e YOLOv8: detecta operadores sem
capacete ou colete dentro de áreas configuráveis, dispara alertas locais, registra evidências com
trilha de auditoria à prova de adulteração e notifica o Técnico de Segurança por e-mail.

**Estado:** Módulos 1 a 4 concluídos.

| Módulo | Pacote | Função |
| --- | --- | --- |
| 1. Visão computacional | `safework/vision/` | Captura, pré-processamento, YOLOv8 + ByteTrack, ROI poligonal |
| 2. Máquina de estados | `safework/logic/`, `notification/` | Timer de 5 s por operador, alerta visual e sonoro |
| 3. Evidência e auditoria | `safework/audit/`, `notification/email.py` | Print, SQLite com hash encadeado, e-mail SMTP assíncrono |
| 4. Interface | `safework/gui/`, `safework/monitoring.py` | Dashboard industrial com vídeo, métricas e ações |

## 1. Instalação

Requer **Python 3.10 ou superior**, com Tkinter (já incluso no instalador oficial do Windows e do
macOS; no Linux: `sudo apt install python3-tk`).

```bash
git clone <url-do-repositorio> SafeWork
cd SafeWork
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Para usar uma GPU NVIDIA, instale antes o PyTorch com CUDA seguindo https://pytorch.org e ajuste
`detector.device: 0` no `config/settings.yaml`.

## 2. Primeira execução (passo a passo)

1. **Modelo:** copie os pesos treinados no Roboflow para `models/safework_ppe.pt`.
   Sem eles, o sistema baixa o `yolov8n.pt`, que só detecta pessoas (veja a seção 6).
2. **Câmera:** em `config/settings.yaml`, ajuste `camera.source` (`0` para a webcam, uma URL
   `rtsp://...` ou um arquivo de vídeo) e os campos `id`, `name` e `area`.
3. **Abrir o dashboard:** `python main.py`.
4. **Desenhar a zona de risco:** clique em **Editar zona de risco** (tecla `E`), marque os
   vértices sobre a imagem, escolha os EPIs exigidos, pressione `Enter` e **Salvar**. A zona passa
   a valer na hora, sem reiniciar.
5. **(Opcional) E-mail:** configure a seção `email:` (seção 7) e teste com
   `python main.py --test-email`.

## 3. O dashboard

```text
┌──────────────────────────────────────────────────────────────────────┐
│ SAFEWORK   CAM-01 · Câmera Prensa 01 · Setor de Prensas  ● AO VIVO   │
├─────────────────────────────────────────────┬────────────────────────┤
│                                             │  CONFORME / ALERTA /   │
│   Vídeo ao vivo com os overlays:            │  INFRAÇÃO              │
│   ROI, caixas, contagem de 5 s, faixas      │  Operadores na zona    │
│                                             │  Em alerta · Infrações │
│                                             │  FPS · Inferência      │
│                                             │  Alarme · E-mail       │
│                                             │  Eventos recentes      │
├─────────────────────────────────────────────┴────────────────────────┤
│ [Editar zona E] [Silenciar alarme M] [Verificar auditoria V]  [Sair Q]│
└──────────────────────────────────────────────────────────────────────┘
```

| Ação | Botão / atalho | O que faz |
| --- | --- | --- |
| Editar zona de risco | `E` | Abre o editor sobre o frame atual. Clique: vértice; botão direito ou `Ctrl+Z`: desfaz; `Enter`: fecha o polígono; `Ctrl+S`: salva; `Esc`: cancela |
| Silenciar alarme | `M` | Liga/desliga só o som. Alertas visuais, evidências e e-mails continuam |
| Verificar auditoria | `V` | Confere a cadeia de hashes e o SHA-256 de cada imagem (roda em segundo plano) |
| Sair | `Q` ou `Esc` | Encerra a câmera e aguarda o envio dos e-mails pendentes |
| Tela cheia | `F11` | Alterna tela cheia |

Indicador de conformidade (do mais grave para o menos grave): **INFRAÇÃO** (vermelho),
**ALERTA** com contagem regressiva (âmbar), **SEM SINAL**, **NÃO VERIFICADO**, **CONFORME** (verde)
e **ÁREA LIVRE**. "Infrações na sessão" conta desde a abertura do programa; "Total na auditoria"
inclui as sessões anteriores.

A interface lê o estado de um serviço que roda em outra thread (`MonitoringService`); por isso ela
continua respondendo mesmo quando a inferência está lenta.

## 4. Linha de comando

```bash
python main.py                                  # dashboard (padrão)
python main.py --source data/teste.mp4          # dashboard com um vídeo gravado
python main.py --source "rtsp://usuario:senha@10.0.0.5/live"
python main.py --mute                           # abre com o alarme silenciado
python main.py --warning-seconds 3              # outro tempo de confirmação

python main.py --cv-window                      # janela OpenCV simples, sem painel
python main.py --source data/teste.mp4 --headless --output logs/saida.mp4   # sem janela, grava vídeo
python main.py --edit-zones                     # editor de zonas OpenCV (sem dashboard)

python main.py --test-email                     # envia um e-mail de teste e sai
python main.py --verify-audit                   # verifica a integridade da auditoria e sai
```

`--output` e `--max-frames` usam automaticamente o modo sem dashboard. Se o Tkinter ou o
CustomTkinter não estiverem disponíveis, o programa avisa e abre a janela OpenCV.

## 5. Testes

```bash
python -m pytest
```

São 128 testes. Os da janela (`tests/test_gui.py`) abrem janelas reais e são pulados
automaticamente se não houver Tkinter ou tela (ex.: servidor sem interface gráfica).

## 6. Modelo de EPI

As classes do modelo são vinculadas pelo **nome**, não pelo índice (`detector.class_aliases`),
porque a ordem varia entre datasets do Roboflow. Classes negativas como `NO-Hardhat` são ignoradas.

Com o `yolov8n.pt` (COCO), só pessoas são detectadas: quem está na zona aparece como
**NÃO VERIFICADO**, nunca como infração, e o painel mostra "Detecção de EPI: Indisponível".

## 7. Configuração (`config/settings.yaml`)

| Seção | Principais campos |
| --- | --- |
| `camera` | `source`, `id`, `name`, `area`, reconexão |
| `preprocessing` | `target_width` (largura de processamento e das evidências), `denoise` |
| `detector` | `weights`, `confidence`, `device`, `tracking`, `class_aliases` |
| `spatial` | `anchor` (`centroid` ou `bottom_center`), faixas do capacete e do colete |
| `alerts` | `warning_seconds` (5), `clear_after_s`, `lost_track_timeout_s`, `blink_hz` |
| `sound` | `enabled`, `backend` (`auto`, `winsound`, `pygame`, `bell`, `none`), tons |
| `evidence` | `directory`, `jpeg_quality` |
| `audit` | `database` |
| `email` | `enabled`, `host`, `port`, `security`, `username`, `sender`, `recipients` |

### E-mail com Gmail

1. Ative a verificação em duas etapas na conta Google e crie uma **senha de app**.
2. Em `email:`, preencha `username`, `sender`, `recipients` e use `enabled: true`.
3. Passe a senha por variável de ambiente, para ela não ir para o GitHub:
   - Windows (PowerShell): `$env:SAFEWORK_SMTP_PASSWORD = "abcd efgh ijkl mnop"`
   - Linux/macOS: `export SAFEWORK_SMTP_PASSWORD="abcd efgh ijkl mnop"`
4. `python main.py --test-email`.

## 8. Funcionamento

### Máquina de estados (Módulo 2)

| Estado | Condição | Alerta |
| --- | --- | --- |
| 0 OK | Conforme, fora da zona ou EPI não verificável | Nenhum |
| 1 Infração inicial | Na zona sem EPI; inicia o timer de 5 s | Caixa e borda piscando vermelho/âmbar, contagem regressiva, bipe intermitente |
| 2 Infração confirmada | Ainda sem EPI após 5 s | Caixa e borda vermelhas, faixa de infração, bipe agudo, evidência e e-mail |

- O timer é por operador (ID do ByteTrack) e usa o tempo do frame.
- Volta ao Estado 0 só após 1 s de conformidade contínua: um frame isolado com capacete não zera o timer.
- Oclusões menores que 2 s mantêm o timer.

### Evidência e auditoria (Módulo 3)

A cada infração confirmada:

1. O frame com overlay é salvo em `evidences/evidencia_CAM01_OP1_20261108_220000.jpg`
   (JPEG 95, cor 4:4:4 para o texto do alerta ficar nítido).
2. Os metadados vão para `data/safework_audit.db` (tabela `violations`): data e hora, câmera,
   origem (IP/URL sem senha), área, zona, operador, tipo de infração, caminho e SHA-256 da imagem.
3. O e-mail com a imagem anexada é enviado por uma thread; o resultado volta para o registro.

Cada registro guarda o hash do anterior. Editar ou apagar uma linha, ou alterar uma imagem, é
detectado pelo botão **Verificar auditoria** ou por `python main.py --verify-audit`.

Consulta rápida: `sqlite3 data/safework_audit.db "SELECT occurred_at, track_id, infraction_type,
email_status FROM violations ORDER BY id DESC LIMIT 10"`.

## 9. Estrutura

```text
SafeWork/
├── main.py                    # linha de comando
├── requirements.txt
├── config/
│   ├── settings.yaml          # toda a configuração
│   └── zones.json             # zonas de risco (coordenadas normalizadas 0-1)
├── safework/
│   ├── app.py                 # composição: monta os objetos e escolhe o modo de execução
│   ├── monitoring.py          # serviço: laço Módulos 1 → 2 → 3 em thread, métricas
│   ├── core/                  # modelos de domínio e configuração
│   ├── vision/                # MÓDULO 1
│   │   ├── capture.py         # webcam/RTSP/arquivo em thread, com reconexão
│   │   ├── preprocessing.py   # redimensionamento e redução de ruído
│   │   ├── detector.py        # YOLOv8 + ByteTrack
│   │   ├── roi.py             # zona poligonal, máscara e persistência
│   │   ├── roi_editor.py      # lógica do editor + editor OpenCV
│   │   ├── spatial.py         # ponto-no-polígono e vínculo de EPI à pessoa
│   │   ├── renderer.py        # overlays
│   │   └── pipeline.py        # orquestração das etapas
│   ├── logic/                 # MÓDULO 2: estados, máquina por operador, eventos
│   ├── audit/                 # MÓDULO 3: evidência, registros, SQLite, auditor
│   ├── notification/          # alerta visual e sonoro; e-mail SMTP assíncrono
│   ├── gui/                   # MÓDULO 4
│   │   ├── dashboard.py       # janela principal, ações e atalhos
│   │   ├── video_panel.py     # feed de vídeo
│   │   ├── metrics_panel.py   # painel lateral de métricas e eventos
│   │   ├── zone_editor.py     # editor de zonas integrado
│   │   ├── viewmodel.py       # lógica de apresentação (sem Tkinter, testável)
│   │   └── theme.py           # cores e fontes
│   └── utils/                 # logging e FPS
├── models/                    # pesos .pt (fora do Git)
├── evidences/  data/  logs/   # evidências, banco de auditoria, logs (fora do Git)
└── tests/                     # 128 testes
```
