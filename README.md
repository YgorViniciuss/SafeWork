# SafeWork

Monitoramento de EPIs em zonas de risco com OpenCV e YOLOv8.
Estado atual: **Módulos 1, 2 e 3 concluídos** (visão computacional; máquina de estados e alertas;
evidência, auditoria e e-mail).

## Estrutura

```text
SafeWork/
├── main.py                  # linha de comando
├── requirements.txt
├── config/
│   ├── settings.yaml        # câmera, modelo, pré-processamento
│   └── zones.json           # zonas de risco (coordenadas normalizadas)
├── safework/
│   ├── app.py               # monta os objetos e conduz o laço de exibição
│   ├── core/                # modelos de domínio e configuração
│   ├── vision/              # MÓDULO 1
│   │   ├── capture.py       # webcam/RTSP/arquivo em thread, com reconexão
│   │   ├── preprocessing.py # redimensionamento e redução de ruído
│   │   ├── detector.py      # YOLOv8 + rastreamento ByteTrack
│   │   ├── roi.py           # zona poligonal, máscara e persistência
│   │   ├── roi_editor.py    # desenho das zonas com o mouse
│   │   ├── spatial.py       # ponto-no-polígono e vínculo de EPI à pessoa
│   │   ├── renderer.py      # overlays
│   │   └── pipeline.py      # orquestração das etapas
│   ├── logic/               # MÓDULO 2
│   │   ├── states.py        # AlertState (0/1/2), AlertEvent, AlertSnapshot
│   │   ├── state_machine.py # máquina de um operador: timer de 5 s e histerese
│   │   ├── engine.py        # uma máquina por ID do ByteTrack
│   │   └── events.py        # barramento de eventos (Observer) e log
│   ├── audit/               # MÓDULO 3
│   │   ├── evidence.py      # print JPEG em evidences/ + SHA-256
│   │   ├── records.py       # metadados da violação e hash encadeado
│   │   ├── repository.py    # trilha de auditoria SQLite e verificação de integridade
│   │   └── auditor.py       # ouvinte de VIOLATION_CONFIRMED: grava, registra e notifica
│   ├── notification/        # alertas locais (Módulo 2) e e-mail (Módulo 3)
│   │   ├── visual.py        # overlay piscante, contagem regressiva, faixa de alerta
│   │   ├── sound.py         # alarme em thread: winsound, pygame ou sinal do terminal
│   │   └── email.py         # SMTP assíncrono: fila + thread, novas tentativas
│   ├── gui/                 # Módulo 5: interface industrial (a implementar)
│   └── utils/               # logging e FPS
├── models/                  # pesos .pt
├── evidences/  logs/  data/
└── tests/
```

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py --edit-zones                    # desenha as zonas sobre um frame da câmera
python main.py                                 # monitora a fonte do settings.yaml (Q ou ESC sai)
python main.py --source data/teste.mp4         # usa um vídeo
python main.py --source "rtsp://usuario:senha@10.0.0.5/live"
python main.py --source data/teste.mp4 --headless --output logs/saida.mp4
python main.py --mute                          # sem alarme sonoro
python main.py --warning-seconds 3             # outro tempo de confirmação
python main.py --test-email                    # envia um e-mail de teste
python main.py --verify-audit                  # confere a integridade da auditoria
python -m pytest                               # testes
```

## Modelo de EPI

Coloque os pesos treinados em `models/safework_ppe.pt`. Sem eles, o sistema baixa o
`yolov8n.pt` (COCO), que só detecta pessoas: quem está na zona aparece como
**EPI não verificado**, nunca como infração.

As classes são vinculadas pelo **nome**, não pelo índice. Se o dataset usar outros nomes,
acrescente-os em `detector.class_aliases` no `settings.yaml`.

## Legenda

| Cor | Situação |
| --- | --- |
| Verde | Na zona, com os EPIs exigidos |
| Vermelho | Na zona, faltando EPI |
| Amarelo | Na zona, EPI não verificável pelo modelo |
| Cinza | Fora da zona |

## Máquina de estados (Módulo 2)

| Estado | Condição | Alerta |
| --- | --- | --- |
| 0 OK | Conforme, fora da zona ou EPI não verificável | Nenhum |
| 1 Infração inicial | Na zona sem EPI; inicia o timer de 5 s | Caixa e borda piscando vermelho/amarelo, contagem regressiva, bipe intermitente |
| 2 Infração confirmada | Ainda sem EPI após 5 s | Caixa e borda vermelhas, faixa de infração, bipe agudo; evento `VIOLATION_CONFIRMED` |

- O timer é por operador (ID do ByteTrack) e usa o tempo do frame.
- Volta ao Estado 0 só após `clear_after_s` (1 s) de conformidade contínua; um frame isolado com capacete não zera o timer.
- Oclusões menores que `lost_track_timeout_s` (2 s) mantêm o timer.
- Para consumir eventos (Módulos 3 e 4): `app.events.subscribe(ouvinte)`, onde o ouvinte tem `handle(event, evidence)`.

## Evidência, auditoria e e-mail (Módulo 3)

A cada infração confirmada:

1. O frame com overlay é salvo em `evidences/evidencia_CAM01_OP1_20261108_220000.jpg`
   (JPEG qualidade 95, cor 4:4:4 para o texto do alerta ficar nítido).
2. Os metadados vão para `data/safework_audit.db` (tabela `violations`): data e hora, câmera,
   origem (IP/URL sem senha), área, zona, operador, tipo de infração, caminho e SHA-256 da imagem.
3. O e-mail com a imagem anexada é enviado por uma thread; o laço de vídeo só enfileira.
   O resultado (`enviado`, `falhou`, `descartado`, `desativado`) volta para o registro.

Cada registro guarda o hash do anterior. Editar ou apagar uma linha, ou alterar uma imagem,
é detectado por `python main.py --verify-audit`.

### Configurando o e-mail (Gmail)

1. Ative a verificação em duas etapas na conta Google e crie uma **senha de app**.
2. No `settings.yaml`, em `email:`, preencha `username`, `sender`, `recipients` e use `enabled: true`.
3. Passe a senha pela variável de ambiente, para não versioná-la:
   - Windows (PowerShell): `$env:SAFEWORK_SMTP_PASSWORD = "abcd efgh ijkl mnop"`
   - Linux/macOS: `export SAFEWORK_SMTP_PASSWORD="abcd efgh ijkl mnop"`
4. Rode `python main.py --test-email`.

Consulta rápida da auditoria: `sqlite3 data/safework_audit.db "SELECT occurred_at, track_id,
infraction_type, email_status FROM violations ORDER BY id DESC LIMIT 10"`.

## Contrato do Módulo 1

`VisionPipeline.stream()` gera um `FrameAnalysis` por frame. Cada `PersonAnalysis` traz
`track_id`, `status`, `zone_id` e `missing_ppe`. O temporizador deve usar `frame.timestamp`,
que funciona tanto ao vivo quanto em vídeo gravado.
