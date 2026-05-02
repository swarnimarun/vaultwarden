use std::{
    collections::HashMap,
    ffi::OsString,
    io::{self, Stdout},
    path::{Path, PathBuf},
    process::exit,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use argon2::{Algorithm as ArgonAlgorithm, Argon2, ParamsBuilder, Version as ArgonVersion};
use crossterm::{
    cursor::Show,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use data_encoding::{BASE64, HEXLOWER};
use openssl::{
    pkey::PKey,
    rsa::{Padding, Rsa},
    symm::{decrypt, Cipher},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use reqwest::header::AUTHORIZATION;
use ring::{
    digest::{Context, SHA256},
    hkdf, hmac, pbkdf2,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    fs,
    io::{AsyncWriteExt, BufWriter},
};
use totp_lite::{totp_custom, Sha1, Sha256, Sha512};

use crate::{Error, MapResult, VERSION};

const DEVICE_TYPE_SDK: &str = "21";
const PBKDF2_OUTPUT_LEN: usize = 32;

pub const DUMP_HELP: &str = "\
    dump   --server <url> --client-id <user.uuid> --client-secret <secret> --out <dir>
                                       Save sync.json, encrypted attachments, and optionally
                                       a decrypted export with decrypted attachments
                                       Flags: --decrypt --master-password <value>
                                              [--encrypted-sync] [--skip-attachments]
                                              [--include-attachment-checksums] [--exclude-domains]
    stats  --server <url> --client-id <user.uuid> --client-secret <secret>
                                       Print aggregate vault stats from /api/tools/stats
                                       or locally decrypted sync data
                                       Flags: [--local-decrypt] [--master-password <value>]
    search --server <url> --client-id <user.uuid> --client-secret <secret> --query <text>
                                       Search locally over decrypted vault contents
                                       Flags: [--domain|--name|--all] [--master-password <value>]
    fetch  --server <url> --client-id <user.uuid> --client-secret <secret> --query <text>
                                       Print or export decrypted item details and attachments
                                       Flags: [--domain|--name|--all] [--json] [--out <dir>]
                                              [--master-password <value>]
    tui    --server <url> --client-id <user.uuid> --client-secret <secret>
                                       Start a simple search-first terminal UI
                                       Flags: [--master-password <value>]
";

#[derive(Clone, Debug)]
struct CommonArgs {
    server: String,
    client_id: String,
    client_secret: String,
    verbose: bool,
}

#[derive(Debug)]
struct DumpArgs {
    common: CommonArgs,
    out_dir: PathBuf,
    exclude_domains: bool,
    encrypted_sync: bool,
    decrypt: bool,
    master_password: Option<String>,
    skip_attachments: bool,
    include_attachment_checksums: bool,
}

#[derive(Debug)]
struct StatsArgs {
    common: CommonArgs,
    local_decrypt: bool,
    master_password: Option<String>,
}

#[derive(Debug)]
struct SearchArgs {
    common: CommonArgs,
    query: String,
    mode: SearchMode,
    master_password: Option<String>,
}

#[derive(Debug)]
struct FetchArgs {
    common: CommonArgs,
    query: String,
    mode: SearchMode,
    master_password: Option<String>,
    json: bool,
    out_dir: Option<PathBuf>,
}

#[derive(Debug)]
struct TuiArgs {
    common: CommonArgs,
    master_password: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchMode {
    Domain,
    Name,
    All,
}

#[derive(Clone, Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(rename = "Key")]
    key: Option<String>,
    #[serde(rename = "PrivateKey")]
    private_key: Option<String>,
    #[serde(rename = "Kdf")]
    kdf: Option<i32>,
    #[serde(rename = "KdfIterations")]
    kdf_iterations: Option<u32>,
    #[serde(rename = "KdfMemory")]
    kdf_memory: Option<u32>,
    #[serde(rename = "KdfParallelism")]
    kdf_parallelism: Option<u32>,
}

#[derive(Clone, Debug)]
struct Session {
    client: reqwest::Client,
    common: CommonArgs,
    token: TokenResponse,
}

#[derive(Clone, Debug)]
struct SyncBundle {
    session: Session,
    sync: Value,
}

#[derive(Clone, Debug)]
struct SymmetricKey {
    enc: Vec<u8>,
    mac: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize)]
struct AttachmentIndexEntry {
    cipher_id: String,
    attachment_id: String,
    source_url: String,
    relative_path: String,
    file_name: Option<String>,
    size: Option<String>,
    checksum_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedAttachmentIndexEntry {
    cipher_id: String,
    attachment_id: String,
    decrypted_file_name: String,
    source_url: String,
    source_path: Option<String>,
    output_path: String,
    mime: Option<String>,
    checksum_sha256: Option<String>,
}

#[derive(Clone, Debug)]
struct AttachmentDownload {
    cipher_id: String,
    attachment_id: String,
    source_url: String,
    relative_path: PathBuf,
    file_name: Option<String>,
    size: Option<String>,
}

#[derive(Clone, Debug)]
struct AccountKeys {
    user_key: Arc<SymmetricKey>,
    org_keys: HashMap<String, Arc<SymmetricKey>>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedFolder {
    id: String,
    name: String,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedField {
    name: Option<String>,
    value: Option<String>,
    #[serde(rename = "type")]
    field_type: Option<i64>,
    linked_id: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedPasswordHistory {
    password: Option<String>,
    last_used_date: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedUri {
    uri: Option<String>,
    match_type: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedAttachment {
    id: String,
    file_name: String,
    size: Option<String>,
    url: String,
    mime: Option<String>,
    #[serde(skip_serializing)]
    key: Option<SymmetricKey>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedLogin {
    username: Option<String>,
    password: Option<String>,
    totp: Option<String>,
    uris: Vec<DecryptedUri>,
    password_history: Vec<DecryptedPasswordHistory>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedCard {
    cardholder_name: Option<String>,
    brand: Option<String>,
    number: Option<String>,
    exp_month: Option<String>,
    exp_year: Option<String>,
    code: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedIdentity {
    title: Option<String>,
    first_name: Option<String>,
    middle_name: Option<String>,
    last_name: Option<String>,
    address1: Option<String>,
    address2: Option<String>,
    address3: Option<String>,
    city: Option<String>,
    state: Option<String>,
    postal_code: Option<String>,
    country: Option<String>,
    company: Option<String>,
    email: Option<String>,
    phone: Option<String>,
    ssn: Option<String>,
    username: Option<String>,
    passport_number: Option<String>,
    license_number: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedSecureNote {
    note_type: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedSshKey {
    private_key: Option<String>,
    public_key: Option<String>,
    fingerprint: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedCipher {
    id: String,
    r#type: i64,
    name: String,
    notes: Option<String>,
    favorite: Option<bool>,
    folder_id: Option<String>,
    organization_id: Option<String>,
    collection_ids: Vec<String>,
    deleted_date: Option<String>,
    creation_date: Option<String>,
    revision_date: Option<String>,
    fields: Vec<DecryptedField>,
    attachments: Vec<DecryptedAttachment>,
    login: Option<DecryptedLogin>,
    card: Option<DecryptedCard>,
    identity: Option<DecryptedIdentity>,
    secure_note: Option<DecryptedSecureNote>,
    ssh_key: Option<DecryptedSshKey>,
    #[serde(skip_serializing)]
    search_blob: String,
    #[serde(skip_serializing)]
    search_domains: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct DecryptedExport {
    encrypted: bool,
    folders: Vec<DecryptedFolder>,
    items: Vec<DecryptedCipher>,
}

#[derive(Clone, Debug, Serialize)]
struct LocalStats {
    total_ciphers: usize,
    ciphers_by_type: HashMap<String, usize>,
    folder_count: usize,
    collection_count: usize,
    send_count: usize,
    attachment_count: usize,
    total_attachment_bytes: u64,
    items_with_attachments: usize,
    trashed_count: usize,
    organization_owned_count: usize,
    personal_owned_count: usize,
}

#[derive(Clone, Debug, Serialize)]
struct FetchResult {
    id: String,
    name: String,
    username: Option<String>,
    password: Option<String>,
    totp: Option<String>,
    notes: Option<String>,
    custom_fields: Vec<DecryptedField>,
    attachments: Vec<FetchAttachmentResult>,
}

#[derive(Clone, Debug, Serialize)]
struct FetchAttachmentResult {
    id: String,
    file_name: String,
    local_path: Option<String>,
    size: Option<String>,
    mime: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TuiFocus {
    Search,
    Results,
    Details,
}

#[derive(Clone, Debug)]
struct TuiState {
    query: String,
    selected: usize,
    focus: TuiFocus,
}

#[derive(Clone, Debug)]
enum TotpAlgorithm {
    Sha1,
    Sha256,
    Sha512,
    Unsupported(String),
}

#[derive(Clone, Debug)]
struct TotpConfig {
    secret: Vec<u8>,
    period: u64,
    digits: u32,
    algorithm: TotpAlgorithm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TotpCode {
    code: String,
    seconds_remaining: u64,
}

pub async fn handle_command(command: &str, pargs: &mut pico_args::Arguments) -> Result<bool, Error> {
    match command {
        "dump" => {
            let args = parse_dump_args(pargs)?;
            ensure_consumed(pargs)?;
            run_dump(args).await?;
            exit(0);
        }
        "stats" => {
            let args = parse_stats_args(pargs)?;
            ensure_consumed(pargs)?;
            run_stats(args).await?;
            exit(0);
        }
        "search" => {
            let args = parse_search_args(pargs)?;
            ensure_consumed(pargs)?;
            run_search(args).await?;
            exit(0);
        }
        "fetch" => {
            let args = parse_fetch_args(pargs)?;
            ensure_consumed(pargs)?;
            run_fetch(args).await?;
            exit(0);
        }
        "tui" => {
            let args = parse_tui_args(pargs)?;
            ensure_consumed(pargs)?;
            run_tui(args).await?;
            exit(0);
        }
        _ => Ok(false),
    }
}

fn ensure_consumed(pargs: &pico_args::Arguments) -> Result<(), Error> {
    let remaining = pargs.clone().finish();
    if remaining.is_empty() {
        Ok(())
    } else {
        err_silent!(format!("Unrecognized arguments: {}", format_os_args(&remaining)))
    }
}

fn parse_common_args(pargs: &mut pico_args::Arguments) -> Result<CommonArgs, Error> {
    let server = pargs
        .opt_value_from_str::<_, String>("--server")
        .map_err(|e| Error::new("Invalid --server argument", e.to_string()))?
        .map(normalize_base_url)
        .map_res("Missing required argument --server")?;
    let client_id = pargs
        .opt_value_from_str::<_, String>("--client-id")
        .map_err(|e| Error::new("Invalid --client-id argument", e.to_string()))?
        .map(normalize_client_id)
        .map_res("Missing required argument --client-id")?;
    let client_secret = pargs
        .opt_value_from_str::<_, String>("--client-secret")
        .map_err(|e| Error::new("Invalid --client-secret argument", e.to_string()))?
        .map_res("Missing required argument --client-secret")?;

    Ok(CommonArgs {
        server,
        client_id,
        client_secret,
        verbose: pargs.contains("--verbose"),
    })
}

fn parse_dump_args(pargs: &mut pico_args::Arguments) -> Result<DumpArgs, Error> {
    let common = parse_common_args(pargs)?;
    let out_dir = pargs
        .opt_value_from_str::<_, PathBuf>("--out")
        .map_err(|e| Error::new("Invalid --out argument", e.to_string()))?
        .map_res("Missing required argument --out")?;

    Ok(DumpArgs {
        common,
        out_dir,
        exclude_domains: pargs.contains("--exclude-domains"),
        encrypted_sync: pargs.contains("--encrypted-sync"),
        decrypt: pargs.contains("--decrypt"),
        master_password: parse_master_password_arg(pargs)?,
        skip_attachments: pargs.contains("--skip-attachments"),
        include_attachment_checksums: pargs.contains("--include-attachment-checksums"),
    })
}

fn parse_stats_args(pargs: &mut pico_args::Arguments) -> Result<StatsArgs, Error> {
    Ok(StatsArgs {
        common: parse_common_args(pargs)?,
        local_decrypt: pargs.contains("--local-decrypt"),
        master_password: parse_master_password_arg(pargs)?,
    })
}

fn parse_search_args(pargs: &mut pico_args::Arguments) -> Result<SearchArgs, Error> {
    let common = parse_common_args(pargs)?;
    let query = pargs
        .opt_value_from_str::<_, String>("--query")
        .map_err(|e| Error::new("Invalid --query argument", e.to_string()))?
        .map_res("Missing required argument --query")?;

    Ok(SearchArgs {
        common,
        query,
        mode: parse_search_mode(pargs),
        master_password: parse_master_password_arg(pargs)?,
    })
}

fn parse_fetch_args(pargs: &mut pico_args::Arguments) -> Result<FetchArgs, Error> {
    let common = parse_common_args(pargs)?;
    let query = pargs
        .opt_value_from_str::<_, String>("--query")
        .map_err(|e| Error::new("Invalid --query argument", e.to_string()))?
        .map_res("Missing required argument --query")?;
    let out_dir = pargs
        .opt_value_from_str::<_, PathBuf>("--out")
        .map_err(|e| Error::new("Invalid --out argument", e.to_string()))?;

    Ok(FetchArgs {
        common,
        query,
        mode: parse_search_mode(pargs),
        master_password: parse_master_password_arg(pargs)?,
        json: pargs.contains("--json"),
        out_dir,
    })
}

fn parse_tui_args(pargs: &mut pico_args::Arguments) -> Result<TuiArgs, Error> {
    Ok(TuiArgs {
        common: parse_common_args(pargs)?,
        master_password: parse_master_password_arg(pargs)?,
    })
}

fn parse_master_password_arg(pargs: &mut pico_args::Arguments) -> Result<Option<String>, Error> {
    pargs
        .opt_value_from_str::<_, String>("--master-password")
        .map_err(|e| Error::new("Invalid --master-password argument", e.to_string()))
}

fn parse_search_mode(pargs: &mut pico_args::Arguments) -> SearchMode {
    if pargs.contains("--domain") {
        SearchMode::Domain
    } else if pargs.contains("--name") {
        SearchMode::Name
    } else {
        let _ = pargs.contains("--all");
        SearchMode::All
    }
}

async fn run_dump(args: DumpArgs) -> Result<(), Error> {
    log_info("dump", &args.common, format!("Starting vault dump from {}", args.common.server));
    fs::create_dir_all(&args.out_dir).await.map_res("Failed to create output directory")?;

    let bundle = login_and_fetch_sync(args.common.clone(), args.exclude_domains).await?;
    let sync_path = args.out_dir.join("sync.json");
    let sync_bytes = serde_json::to_vec_pretty(&bundle.sync).map_res("Failed to serialize sync JSON")?;
    fs::write(&sync_path, sync_bytes).await.map_res("Failed to write sync.json")?;
    log_info("dump", &args.common, format!("Wrote {}", sync_path.display()));

    let attachment_index = if args.skip_attachments {
        Vec::new()
    } else {
        download_raw_attachments(
            &bundle.session.client,
            &args.common,
            &bundle.sync,
            &args.out_dir,
            args.include_attachment_checksums,
        )
        .await?
    };

    if args.encrypted_sync || !args.decrypt {
        let raw_index_path = args.out_dir.join("attachments-index.json");
        let raw_index_bytes =
            serde_json::to_vec_pretty(&attachment_index).map_res("Failed to serialize attachments index")?;
        fs::write(&raw_index_path, raw_index_bytes).await.map_res("Failed to write attachments-index.json")?;
        log_info("dump", &args.common, format!("Wrote {}", raw_index_path.display()));
    }

    if args.decrypt {
        let master_password = resolve_master_password(args.master_password, "Master password: ")?;
        let keys = derive_account_keys(&bundle, &master_password, &args.common)?;
        let export = decrypt_export(&bundle.sync, &keys, &args.common)?;
        let decrypt_root = args.out_dir.join("decrypted");
        fs::create_dir_all(&decrypt_root).await.map_res("Failed to create decrypted output directory")?;

        let vault_path = decrypt_root.join("vault.json");
        let vault_bytes = serde_json::to_vec_pretty(&export).map_res("Failed to serialize decrypted vault export")?;
        fs::write(&vault_path, vault_bytes).await.map_res("Failed to write decrypted vault.json")?;
        log_info("dump", &args.common, format!("Wrote {}", vault_path.display()));

        let decrypted_attachment_index = if args.skip_attachments {
            Vec::new()
        } else {
            decrypt_export_attachments(
                &bundle.session.client,
                &args.common,
                &export.items,
                &decrypt_root,
                args.include_attachment_checksums,
            )
            .await?
        };

        let decrypted_index_path = decrypt_root.join("attachments-index.json");
        let decrypted_index_bytes = serde_json::to_vec_pretty(&decrypted_attachment_index)
            .map_res("Failed to serialize decrypted attachment index")?;
        fs::write(&decrypted_index_path, decrypted_index_bytes)
            .await
            .map_res("Failed to write decrypted attachments-index.json")?;
        log_info("dump", &args.common, format!("Wrote {}", decrypted_index_path.display()));
    }

    Ok(())
}

async fn run_stats(args: StatsArgs) -> Result<(), Error> {
    if !args.local_decrypt {
        let session = login_with_api_key(args.common.clone()).await?;
        let stats = fetch_server_stats(&session).await?;
        println!("{}", serde_json::to_string_pretty(&stats).map_res("Failed to format server stats response")?);
        return Ok(());
    }

    let bundle = login_and_fetch_sync(args.common.clone(), false).await?;
    let master_password = resolve_master_password(args.master_password, "Master password: ")?;
    let keys = derive_account_keys(&bundle, &master_password, &args.common)?;
    let export = decrypt_export(&bundle.sync, &keys, &args.common)?;
    let stats = compute_local_stats(&bundle.sync, &export);
    println!("{}", serde_json::to_string_pretty(&stats).map_res("Failed to format local stats response")?);
    Ok(())
}

async fn run_search(args: SearchArgs) -> Result<(), Error> {
    let export = load_decrypted_export(args.common.clone(), args.master_password).await?;
    let matches = search_items(&export.items, &args.query, args.mode);
    for item in matches {
        let username = item.login.as_ref().and_then(|login| login.username.clone()).unwrap_or_default();
        let domains = if item.search_domains.is_empty() {
            String::new()
        } else {
            format!(" [{}]", item.search_domains.join(", "))
        };
        println!("{}  {}  {}{}", item.id, item.name, username, domains);
    }
    Ok(())
}

async fn run_fetch(args: FetchArgs) -> Result<(), Error> {
    let bundle = login_and_fetch_sync(args.common.clone(), false).await?;
    let master_password = resolve_master_password(args.master_password, "Master password: ")?;
    let keys = derive_account_keys(&bundle, &master_password, &args.common)?;
    let export = decrypt_export(&bundle.sync, &keys, &args.common)?;
    let matches = search_items(&export.items, &args.query, args.mode);
    let out_dir = args.out_dir.unwrap_or_else(default_fetch_dir);

    let mut results = Vec::new();
    for item in matches {
        let attachments = if item.attachments.is_empty() {
            Vec::new()
        } else {
            fetch_item_attachments(&bundle.session.client, &args.common, item, &out_dir).await?
        };
        results.push(FetchResult {
            id: item.id.clone(),
            name: item.name.clone(),
            username: item.login.as_ref().and_then(|login| login.username.clone()),
            password: item.login.as_ref().and_then(|login| login.password.clone()),
            totp: item.login.as_ref().and_then(|login| login.totp.clone()),
            notes: item.notes.clone(),
            custom_fields: item.fields.clone(),
            attachments,
        });
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results).map_res("Failed to format fetch JSON output")?);
    } else {
        for result in results {
            println!("id: {}", result.id);
            println!("name: {}", result.name);
            if let Some(username) = result.username {
                println!("username: {}", username);
            }
            if let Some(password) = result.password {
                println!("password: {}", password);
            }
            if let Some(totp) = result.totp {
                println!("totp: {}", totp);
            }
            if let Some(notes) = result.notes {
                println!("notes: {}", notes);
            }
            for field in result.custom_fields {
                println!("field: {}={}", field.name.unwrap_or_default(), field.value.unwrap_or_default());
            }
            for attachment in result.attachments {
                println!("attachment: {} {}", attachment.file_name, attachment.local_path.unwrap_or_default());
            }
            println!();
        }
    }

    Ok(())
}

async fn run_tui(args: TuiArgs) -> Result<(), Error> {
    let export = load_decrypted_export(args.common.clone(), args.master_password).await?;
    let mut terminal = setup_tui_terminal()?;
    let _guard = TuiTerminalGuard;

    let mut state = TuiState {
        query: String::new(),
        selected: 0,
        focus: TuiFocus::Search,
    };

    loop {
        let matches = search_items(&export.items, &state.query, SearchMode::All);
        if state.selected >= matches.len() && !matches.is_empty() {
            state.selected = matches.len() - 1;
        } else if matches.is_empty() {
            state.selected = 0;
        }

        terminal.draw(|frame| render_tui(frame, &state, &matches)).map_res("Failed to render terminal UI")?;

        if event::poll(Duration::from_millis(250)).map_res("Failed to poll terminal events")? {
            let Event::Key(key) = event::read().map_res("Failed to read terminal event")? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if handle_tui_key_event(&mut state, matches.len(), key) {
                break;
            }
        }
    }

    Ok(())
}

fn setup_tui_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>, Error> {
    enable_raw_mode().map_res("Failed to enable raw terminal mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).map_res("Failed to enter alternate terminal screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_res("Failed to initialize terminal backend")?;
    terminal.hide_cursor().map_res("Failed to hide terminal cursor")?;
    Ok(terminal)
}

struct TuiTerminalGuard;

impl Drop for TuiTerminalGuard {
    fn drop(&mut self) {
        drop(disable_raw_mode());
        let mut stdout = io::stdout();
        drop(execute!(stdout, Show, LeaveAlternateScreen));
    }
}

fn handle_tui_key_event(state: &mut TuiState, match_count: usize, key: KeyEvent) -> bool {
    if matches!(key.code, KeyCode::Char('q')) && !matches!(state.focus, TuiFocus::Search) {
        return true;
    }
    if key.code == KeyCode::Esc {
        state.focus = TuiFocus::Results;
        return false;
    }
    if key.code == KeyCode::Char('/') {
        state.focus = TuiFocus::Search;
        return false;
    }

    match state.focus {
        TuiFocus::Search => match key.code {
            KeyCode::Enter => state.focus = TuiFocus::Results,
            KeyCode::Backspace => {
                state.query.pop();
                state.selected = 0;
            }
            KeyCode::Char(ch) if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT => {
                state.query.push(ch);
                state.selected = 0;
            }
            KeyCode::Up | KeyCode::Char('k') => move_selection_up(state),
            KeyCode::Down | KeyCode::Char('j') => move_selection_down(state, match_count),
            _ => {}
        },
        TuiFocus::Results => match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Up | KeyCode::Char('k') => move_selection_up(state),
            KeyCode::Down | KeyCode::Char('j') => move_selection_down(state, match_count),
            KeyCode::Enter | KeyCode::Right => state.focus = TuiFocus::Details,
            _ => {}
        },
        TuiFocus::Details => match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Left | KeyCode::Enter => state.focus = TuiFocus::Results,
            KeyCode::Up | KeyCode::Char('k') => move_selection_up(state),
            KeyCode::Down | KeyCode::Char('j') => move_selection_down(state, match_count),
            _ => {}
        },
    }

    false
}

fn move_selection_up(state: &mut TuiState) {
    state.selected = state.selected.saturating_sub(1);
}

fn move_selection_down(state: &mut TuiState, match_count: usize) {
    if match_count > 0 {
        state.selected = (state.selected + 1).min(match_count - 1);
    }
}

fn render_tui(frame: &mut Frame<'_>, state: &TuiState, matches: &[&DecryptedCipher]) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(frame.size());
    render_search_box(frame, root[0], state);

    let body = if root[1].width >= 120 {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
            .split(root[1])
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(root[1])
    };

    render_results_list(frame, body[0], state, matches);
    render_details(frame, body[1], state, matches.get(state.selected).copied());
}

fn render_search_box(frame: &mut Frame<'_>, area: Rect, state: &TuiState) {
    let title = match state.focus {
        TuiFocus::Search => "Search [focused]",
        _ => "Search",
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let paragraph = Paragraph::new(state.query.as_str()).block(block).style(match state.focus {
        TuiFocus::Search => Style::default().fg(Color::Cyan),
        _ => Style::default(),
    });
    frame.render_widget(paragraph, area);

    if state.focus == TuiFocus::Search {
        let cursor_x = area.x.saturating_add(state.query.chars().count() as u16 + 1);
        let cursor_y = area.y.saturating_add(1);
        frame.set_cursor(cursor_x.min(area.right().saturating_sub(1)), cursor_y);
    }
}

fn render_results_list(frame: &mut Frame<'_>, area: Rect, state: &TuiState, matches: &[&DecryptedCipher]) {
    let title = format!("Results ({})", matches.len());
    let items = if matches.is_empty() {
        vec![ListItem::new("No matches")]
    } else {
        matches
            .iter()
            .map(|item| {
                let subtitle = item
                    .login
                    .as_ref()
                    .and_then(|login| login.username.as_deref())
                    .or_else(|| item.search_domains.first().map(String::as_str))
                    .unwrap_or("");
                let line = if subtitle.is_empty() {
                    item.name.clone()
                } else {
                    format!("{}  [{}]", item.name, subtitle)
                };
                ListItem::new(line)
            })
            .collect()
    };
    let mut list_state = ListState::default().with_selected((!matches.is_empty()).then_some(state.selected));
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(match state.focus {
            TuiFocus::Results => format!("{title} [focused]"),
            _ => title,
        }))
        .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");
    frame.render_stateful_widget(list, area, &mut list_state);
}

fn render_details(frame: &mut Frame<'_>, area: Rect, state: &TuiState, item: Option<&DecryptedCipher>) {
    let title = match state.focus {
        TuiFocus::Details => "Details [focused]",
        _ => "Details",
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(Clear, inner);

    let Some(item) = item else {
        return;
    };

    let lines = build_item_detail_lines(item);
    let paragraph = Paragraph::new(lines).wrap(Wrap {
        trim: false,
    });
    frame.render_widget(paragraph, inner);
}

fn build_item_detail_lines(item: &DecryptedCipher) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("name: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(item.name.clone()),
        ]),
        Line::from(vec![
            Span::styled("type: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(cipher_type_name(item.r#type)),
        ]),
        Line::from(vec![
            Span::styled("id: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(item.id.clone()),
        ]),
    ];

    if let Some(notes) = &item.notes {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled("notes", Style::default().add_modifier(Modifier::BOLD))));
        for line in notes.lines() {
            lines.push(Line::from(line.to_owned()));
        }
    }

    if let Some(login) = &item.login {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled("login", Style::default().add_modifier(Modifier::BOLD))));
        push_optional_line(&mut lines, "username", login.username.clone());
        push_optional_line(&mut lines, "password", login.password.clone());
        lines.extend(render_totp_lines(login.totp.as_deref()));
        for uri in &login.uris {
            push_optional_line(&mut lines, "uri", uri.uri.clone());
        }
    }

    if !item.fields.is_empty() {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled("fields", Style::default().add_modifier(Modifier::BOLD))));
        for field in &item.fields {
            let name = field.name.clone().unwrap_or_else(|| "(unnamed)".to_owned());
            let value = field.value.clone().unwrap_or_default();
            lines.push(Line::from(format!("{name}: {value}")));
        }
    }

    if !item.attachments.is_empty() {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled("attachments", Style::default().add_modifier(Modifier::BOLD))));
        for attachment in &item.attachments {
            lines.push(Line::from(attachment.file_name.clone()));
        }
    }

    lines
}

fn push_optional_line(lines: &mut Vec<Line<'static>>, label: &str, value: Option<String>) {
    if let Some(value) = value {
        lines.push(Line::from(format!("{label}: {value}")));
    }
}

fn render_totp_lines(value: Option<&str>) -> Vec<Line<'static>> {
    let Some(value) = value else {
        return Vec::new();
    };

    match parse_totp_value(value) {
        Ok(config) => match generate_totp_code(&config, unix_timestamp_now()) {
            Ok(code) => vec![
                Line::from(format!("totp: {}", code.code)),
                Line::from(format!("refreshes in: {}s", code.seconds_remaining)),
            ],
            Err(err) => vec![Line::from(format!("totp: {err}"))],
        },
        Err(err) => vec![Line::from(format!("totp: {err}"))],
    }
}

async fn login_and_fetch_sync(common: CommonArgs, exclude_domains: bool) -> Result<SyncBundle, Error> {
    let session = login_with_api_key(common).await?;
    let sync = fetch_sync(&session, exclude_domains).await?;
    Ok(SyncBundle {
        session,
        sync,
    })
}

async fn login_with_api_key(common: CommonArgs) -> Result<Session, Error> {
    let client = reqwest::Client::builder().no_proxy().build().map_res("Failed to create HTTP client")?;
    let login_url = format!("{}/identity/connect/token", common.server);
    let device_identifier =
        format!("vaultwarden-cli-{}", common.client_id.strip_prefix("user.").unwrap_or(common.client_id.as_str()));
    let device_name = format!("vaultwarden-cli/{}", VERSION.unwrap_or("dev"));
    log_info("auth", &common, format!("Authenticating with {}", login_url));

    let response = client
        .post(login_url)
        .form(&[
            ("grant_type", "client_credentials"),
            ("scope", "api"),
            ("client_id", common.client_id.as_str()),
            ("client_secret", common.client_secret.as_str()),
            ("device_identifier", device_identifier.as_str()),
            ("device_name", device_name.as_str()),
            ("device_type", DEVICE_TYPE_SDK),
        ])
        .send()
        .await
        .map_res("Failed to authenticate with API key")?;
    let response = ensure_success_response(response, "Vault authentication request failed", "auth", &common).await?;
    let token: TokenResponse = response.json().await.map_res("Failed to decode token response")?;

    Ok(Session {
        client,
        common,
        token,
    })
}

async fn fetch_sync(session: &Session, exclude_domains: bool) -> Result<Value, Error> {
    let sync_url = format!(
        "{}/api/sync?excludeDomains={}",
        session.common.server,
        if exclude_domains {
            "true"
        } else {
            "false"
        }
    );
    log_info("sync", &session.common, format!("Fetching sync payload from {}", sync_url));

    let response = session
        .client
        .get(sync_url)
        .header(AUTHORIZATION, format!("Bearer {}", session.token.access_token))
        .send()
        .await
        .map_res("Failed to fetch vault sync data")?;

    ensure_success_response(response, "Vault sync request failed", "sync", &session.common)
        .await?
        .json()
        .await
        .map_res("Failed to decode vault sync response")
}

async fn fetch_server_stats(session: &Session) -> Result<Value, Error> {
    let stats_url = format!("{}/api/tools/stats", session.common.server);
    log_info("stats", &session.common, format!("Fetching aggregate stats from {}", stats_url));

    let response = session
        .client
        .get(stats_url)
        .header(AUTHORIZATION, format!("Bearer {}", session.token.access_token))
        .send()
        .await
        .map_res("Failed to fetch server stats")?;

    ensure_success_response(response, "Vault server stats request failed", "stats", &session.common)
        .await?
        .json()
        .await
        .map_res("Failed to decode server stats response")
}

async fn download_raw_attachments(
    client: &reqwest::Client,
    common: &CommonArgs,
    sync_json: &Value,
    out_dir: &Path,
    include_checksums: bool,
) -> Result<Vec<AttachmentIndexEntry>, Error> {
    let attachments = collect_attachment_downloads(sync_json);
    let attachment_root = out_dir.join("attachments");
    fs::create_dir_all(&attachment_root).await.map_res("Failed to create attachments output directory")?;

    let mut index = Vec::with_capacity(attachments.len());
    for attachment in attachments {
        let destination = out_dir.join(&attachment.relative_path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).await.map_res("Failed to create attachment directory")?;
        }

        let checksum = download_file(client, common, &attachment.source_url, &destination, include_checksums).await?;
        index.push(AttachmentIndexEntry {
            cipher_id: attachment.cipher_id,
            attachment_id: attachment.attachment_id,
            source_url: attachment.source_url,
            relative_path: path_to_forward_slashes(&attachment.relative_path),
            file_name: attachment.file_name,
            size: attachment.size,
            checksum_sha256: checksum,
        });
    }

    Ok(index)
}

async fn decrypt_export_attachments(
    client: &reqwest::Client,
    common: &CommonArgs,
    items: &[DecryptedCipher],
    decrypt_root: &Path,
    include_checksums: bool,
) -> Result<Vec<DecryptedAttachmentIndexEntry>, Error> {
    let mut index = Vec::new();

    for item in items {
        for attachment in &item.attachments {
            let Some(key) = attachment.key.clone() else {
                return Err(Error::new(
                    "Missing attachment key",
                    format!("cipher={} attachment={}", item.id, attachment.id),
                ));
            };

            let cipher_dir = decrypt_root.join("attachments").join(&item.id);
            fs::create_dir_all(&cipher_dir).await.map_res("Failed to create decrypted attachment directory")?;
            let output_path = cipher_dir.join(sanitize_file_name(&attachment.file_name));
            let encrypted_path = decrypt_root.join("tmp").join(&item.id).join(&attachment.id);
            if let Some(parent) = encrypted_path.parent() {
                fs::create_dir_all(parent).await.map_res("Failed to create temporary attachment directory")?;
            }

            download_file(client, common, &attachment.url, &encrypted_path, false).await?;
            let (checksum, _) = decrypt_attachment_file(&encrypted_path, &output_path, &key, include_checksums).await?;
            index.push(DecryptedAttachmentIndexEntry {
                cipher_id: item.id.clone(),
                attachment_id: attachment.id.clone(),
                decrypted_file_name: attachment.file_name.clone(),
                source_url: attachment.url.clone(),
                source_path: Some(encrypted_path.display().to_string()),
                output_path: output_path.display().to_string(),
                mime: attachment.mime.clone(),
                checksum_sha256: checksum,
            });
        }
    }

    Ok(index)
}

async fn fetch_item_attachments(
    client: &reqwest::Client,
    common: &CommonArgs,
    item: &DecryptedCipher,
    out_dir: &Path,
) -> Result<Vec<FetchAttachmentResult>, Error> {
    let mut results = Vec::new();
    let item_dir = out_dir.join(&item.id);
    fs::create_dir_all(&item_dir).await.map_res("Failed to create fetch attachment output directory")?;

    for attachment in &item.attachments {
        let local_path = if let Some(key) = attachment.key.clone() {
            let encrypted_path = item_dir.join(format!("{}.enc", attachment.id));
            download_file(client, common, &attachment.url, &encrypted_path, false).await?;
            let decrypted_path = item_dir.join(sanitize_file_name(&attachment.file_name));
            decrypt_attachment_file(&encrypted_path, &decrypted_path, &key, false).await?;
            Some(decrypted_path.display().to_string())
        } else {
            None
        };

        results.push(FetchAttachmentResult {
            id: attachment.id.clone(),
            file_name: attachment.file_name.clone(),
            local_path,
            size: attachment.size.clone(),
            mime: attachment.mime.clone(),
        });
    }

    Ok(results)
}

async fn decrypt_attachment_file(
    encrypted_path: &Path,
    output_path: &Path,
    key: &SymmetricKey,
    include_checksum: bool,
) -> Result<(Option<String>, usize), Error> {
    let bytes = fs::read(encrypted_path).await.map_res("Failed to read encrypted attachment file")?;
    let decrypted = decrypt_bytes_with_key(&bytes, key)?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).await.map_res("Failed to create decrypted attachment output directory")?;
    }
    fs::write(output_path, &decrypted).await.map_res("Failed to write decrypted attachment file")?;

    let checksum = include_checksum.then(|| {
        let mut digest = Context::new(&SHA256);
        digest.update(&decrypted);
        HEXLOWER.encode(digest.finish().as_ref())
    });
    Ok((checksum, decrypted.len()))
}

async fn download_file(
    client: &reqwest::Client,
    common: &CommonArgs,
    source_url: &str,
    destination: &Path,
    include_checksum: bool,
) -> Result<Option<String>, Error> {
    let mut response = client.get(source_url).send().await.map_res("Failed to download attachment")?;
    response = ensure_success_response(response, "Attachment download request failed", "dump", common).await?;

    let file = fs::File::create(destination).await.map_res("Failed to create attachment output file")?;
    let mut writer = BufWriter::new(file);
    let mut digest = include_checksum.then(|| Context::new(&SHA256));

    while let Some(chunk) = response.chunk().await.map_res("Failed while streaming attachment body")? {
        if let Some(digest) = digest.as_mut() {
            digest.update(&chunk);
        }
        writer.write_all(&chunk).await.map_res("Failed writing attachment to disk")?;
    }
    writer.flush().await.map_res("Failed to flush attachment file")?;

    Ok(digest.map(|digest| HEXLOWER.encode(digest.finish().as_ref())))
}

fn collect_attachment_downloads(sync_json: &Value) -> Vec<AttachmentDownload> {
    let Some(ciphers) = sync_json.get("ciphers").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut attachments = Vec::new();
    for cipher in ciphers {
        let Some(cipher_id) = cipher.get("id").and_then(Value::as_str) else {
            continue;
        };
        let Some(cipher_attachments) = cipher.get("attachments").and_then(Value::as_array) else {
            continue;
        };

        for attachment in cipher_attachments {
            let Some(attachment_id) = attachment.get("id").and_then(Value::as_str) else {
                continue;
            };
            let Some(source_url) = attachment.get("url").and_then(Value::as_str) else {
                continue;
            };

            attachments.push(AttachmentDownload {
                cipher_id: cipher_id.to_owned(),
                attachment_id: attachment_id.to_owned(),
                source_url: source_url.to_owned(),
                relative_path: PathBuf::from("attachments").join(cipher_id).join(attachment_id),
                file_name: attachment.get("fileName").and_then(Value::as_str).map(str::to_owned),
                size: attachment.get("size").and_then(Value::as_str).map(str::to_owned),
            });
        }
    }

    attachments
}

fn derive_account_keys(bundle: &SyncBundle, master_password: &str, common: &CommonArgs) -> Result<AccountKeys, Error> {
    let profile = bundle.sync.get("profile").map_res("Sync response missing profile")?;
    let email = profile
        .get("email")
        .and_then(Value::as_str)
        .map(normalize_email)
        .map_res("Sync response missing profile.email")?;
    let key_cipher = profile
        .get("key")
        .and_then(Value::as_str)
        .or(bundle.session.token.key.as_deref())
        .map_res("Missing encrypted user key")?;

    let master_key = derive_master_key(master_password, &email, &bundle.session.token)?;
    let stretched_master_keys = stretch_master_key_candidates(&master_key)?;
    let user_key_bytes = decrypt_user_key_with_candidates(key_cipher, &stretched_master_keys, common)?;
    let user_key = Arc::new(SymmetricKey::from_bytes(user_key_bytes)?);

    let private_key =
        match profile.get("privateKey").and_then(Value::as_str).or(bundle.session.token.private_key.as_deref()) {
            Some(private_key_cipher) => {
                let private_key_bytes = decrypt_cipher_string_to_bytes(private_key_cipher, &user_key, None)?;
                let rsa = parse_decrypted_private_key(&private_key_bytes)?;
                Some(Arc::new(rsa))
            }
            None => None,
        };

    let mut org_keys = HashMap::new();
    if let (Some(orgs), Some(private_key)) =
        (profile.get("organizations").and_then(Value::as_array), private_key.as_ref())
    {
        for org in orgs {
            let Some(org_id) = org.get("id").and_then(Value::as_str) else {
                continue;
            };
            let Some(org_key_cipher) = org.get("key").and_then(Value::as_str) else {
                continue;
            };
            match decrypt_rsa_cipher_string_to_bytes(org_key_cipher, private_key) {
                Ok(key_bytes) => {
                    let key = SymmetricKey::from_bytes(key_bytes)?;
                    org_keys.insert(org_id.to_owned(), Arc::new(key));
                }
                Err(err) => {
                    log_verbose("decrypt", common, format!("Skipping org key for {}: {}", org_id, err));
                }
            }
        }
    }

    Ok(AccountKeys {
        user_key,
        org_keys,
    })
}

fn decrypt_export(sync: &Value, keys: &AccountKeys, common: &CommonArgs) -> Result<DecryptedExport, Error> {
    let folders = sync
        .get("folders")
        .and_then(Value::as_array)
        .map(|folders| {
            folders
                .iter()
                .filter_map(|folder| {
                    let id = folder.get("id").and_then(Value::as_str)?;
                    let name_enc = folder.get("name").and_then(Value::as_str)?;
                    let name = decrypt_cipher_string_to_string(name_enc, &keys.user_key, None).ok()?;
                    Some(DecryptedFolder {
                        id: id.to_owned(),
                        name,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut items = Vec::new();
    for cipher in sync.get("ciphers").and_then(Value::as_array).into_iter().flatten() {
        items.push(decrypt_cipher(cipher, keys, common)?);
    }

    Ok(DecryptedExport {
        encrypted: false,
        folders,
        items,
    })
}

fn decrypt_cipher(cipher: &Value, keys: &AccountKeys, common: &CommonArgs) -> Result<DecryptedCipher, Error> {
    let id = get_required_str(cipher, "id")?.to_owned();
    let cipher_type = cipher.get("type").and_then(Value::as_i64).map_res("Cipher missing type")?;
    let organization_id = cipher.get("organizationId").and_then(Value::as_str).map(str::to_owned);
    let owner_key = match organization_id.as_deref() {
        Some(org_id) => keys.org_keys.get(org_id).cloned().unwrap_or_else(|| keys.user_key.clone()),
        None => keys.user_key.clone(),
    };
    let item_key = match cipher.get("key").and_then(Value::as_str) {
        Some(enc) => Arc::new(SymmetricKey::from_bytes(decrypt_cipher_string_to_bytes(enc, &owner_key, None)?)?),
        None => owner_key,
    };

    let name = decrypt_optional_string_field(cipher, "name", &item_key)?.unwrap_or_default();
    let notes = decrypt_optional_string_field(cipher, "notes", &item_key)?;
    let fields = decrypt_fields(cipher.get("fields"), &item_key)?;
    let attachments = decrypt_attachments(cipher.get("attachments"), &item_key, common)?;
    let collection_ids = cipher
        .get("collectionIds")
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).map(str::to_owned).collect::<Vec<_>>())
        .unwrap_or_default();

    let login = decrypt_login(cipher.get("login"), cipher.get("passwordHistory"), &item_key)?;
    let card = decrypt_card(cipher.get("card"), &item_key)?;
    let identity = decrypt_identity(cipher.get("identity"), &item_key)?;
    let secure_note = decrypt_secure_note(cipher.get("secureNote"))?;
    let ssh_key = decrypt_ssh_key(cipher.get("sshKey"), &item_key)?;

    let mut domains = Vec::new();
    let mut search_parts = vec![name.clone()];
    if let Some(notes) = &notes {
        search_parts.push(notes.clone());
    }
    for field in &fields {
        if let Some(name) = &field.name {
            search_parts.push(name.clone());
        }
        if let Some(value) = &field.value {
            search_parts.push(value.clone());
        }
    }
    if let Some(login) = &login {
        if let Some(username) = &login.username {
            search_parts.push(username.clone());
        }
        for uri in &login.uris {
            if let Some(uri_value) = &uri.uri {
                search_parts.push(uri_value.clone());
                if let Some(domain) = normalize_uri_host(uri_value) {
                    domains.push(domain);
                }
            }
        }
        for history in &login.password_history {
            if let Some(password) = &history.password {
                search_parts.push(password.clone());
            }
        }
    }
    if let Some(identity) = &identity {
        if let Some(email) = &identity.email {
            search_parts.push(email.clone());
        }
        if let Some(username) = &identity.username {
            search_parts.push(username.clone());
        }
    }

    Ok(DecryptedCipher {
        id,
        r#type: cipher_type,
        name,
        notes,
        favorite: cipher.get("favorite").and_then(Value::as_bool),
        folder_id: cipher.get("folderId").and_then(Value::as_str).map(str::to_owned),
        organization_id,
        collection_ids,
        deleted_date: cipher.get("deletedDate").and_then(Value::as_str).map(str::to_owned),
        creation_date: cipher.get("creationDate").and_then(Value::as_str).map(str::to_owned),
        revision_date: cipher.get("revisionDate").and_then(Value::as_str).map(str::to_owned),
        fields,
        attachments,
        login,
        card,
        identity,
        secure_note,
        ssh_key,
        search_blob: search_parts.join("\n").to_lowercase(),
        search_domains: domains,
    })
}

fn decrypt_fields(value: Option<&Value>, key: &SymmetricKey) -> Result<Vec<DecryptedField>, Error> {
    let Some(fields) = value.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    fields
        .iter()
        .map(|field| {
            Ok(DecryptedField {
                name: decrypt_value_string(field.get("name"), key)?,
                value: decrypt_value_string(field.get("value"), key)?,
                field_type: field.get("type").and_then(Value::as_i64),
                linked_id: field.get("linkedId").and_then(Value::as_i64),
            })
        })
        .collect()
}

fn decrypt_login(
    value: Option<&Value>,
    history_value: Option<&Value>,
    key: &SymmetricKey,
) -> Result<Option<DecryptedLogin>, Error> {
    let Some(login) = value else {
        return Ok(None);
    };

    let uris = login
        .get("uris")
        .and_then(Value::as_array)
        .map(|uris| {
            uris.iter()
                .map(|uri| {
                    Ok(DecryptedUri {
                        uri: decrypt_value_string(uri.get("uri"), key)?,
                        match_type: uri.get("match").and_then(Value::as_i64),
                    })
                })
                .collect::<Result<Vec<_>, Error>>()
        })
        .transpose()?
        .unwrap_or_default();

    let password_history = history_value
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .map(|entry| {
                    Ok(DecryptedPasswordHistory {
                        password: decrypt_value_string(entry.get("password"), key)?,
                        last_used_date: entry.get("lastUsedDate").and_then(Value::as_str).map(str::to_owned),
                    })
                })
                .collect::<Result<Vec<_>, Error>>()
        })
        .transpose()?
        .unwrap_or_default();

    Ok(Some(DecryptedLogin {
        username: decrypt_value_string(login.get("username"), key)?,
        password: decrypt_value_string(login.get("password"), key)?,
        totp: decrypt_value_string(login.get("totp"), key)?,
        uris,
        password_history,
    }))
}

fn decrypt_card(value: Option<&Value>, key: &SymmetricKey) -> Result<Option<DecryptedCard>, Error> {
    let Some(card) = value else {
        return Ok(None);
    };

    Ok(Some(DecryptedCard {
        cardholder_name: decrypt_value_string(card.get("cardholderName"), key)?,
        brand: decrypt_value_string(card.get("brand"), key)?,
        number: decrypt_value_string(card.get("number"), key)?,
        exp_month: decrypt_value_string(card.get("expMonth"), key)?,
        exp_year: decrypt_value_string(card.get("expYear"), key)?,
        code: decrypt_value_string(card.get("code"), key)?,
    }))
}

fn decrypt_identity(value: Option<&Value>, key: &SymmetricKey) -> Result<Option<DecryptedIdentity>, Error> {
    let Some(identity) = value else {
        return Ok(None);
    };

    Ok(Some(DecryptedIdentity {
        title: decrypt_value_string(identity.get("title"), key)?,
        first_name: decrypt_value_string(identity.get("firstName"), key)?,
        middle_name: decrypt_value_string(identity.get("middleName"), key)?,
        last_name: decrypt_value_string(identity.get("lastName"), key)?,
        address1: decrypt_value_string(identity.get("address1"), key)?,
        address2: decrypt_value_string(identity.get("address2"), key)?,
        address3: decrypt_value_string(identity.get("address3"), key)?,
        city: decrypt_value_string(identity.get("city"), key)?,
        state: decrypt_value_string(identity.get("state"), key)?,
        postal_code: decrypt_value_string(identity.get("postalCode"), key)?,
        country: decrypt_value_string(identity.get("country"), key)?,
        company: decrypt_value_string(identity.get("company"), key)?,
        email: decrypt_value_string(identity.get("email"), key)?,
        phone: decrypt_value_string(identity.get("phone"), key)?,
        ssn: decrypt_value_string(identity.get("ssn"), key)?,
        username: decrypt_value_string(identity.get("username"), key)?,
        passport_number: decrypt_value_string(identity.get("passportNumber"), key)?,
        license_number: decrypt_value_string(identity.get("licenseNumber"), key)?,
    }))
}

fn decrypt_secure_note(value: Option<&Value>) -> Result<Option<DecryptedSecureNote>, Error> {
    let Some(note) = value else {
        return Ok(None);
    };
    Ok(Some(DecryptedSecureNote {
        note_type: note.get("type").and_then(Value::as_i64),
    }))
}

fn decrypt_ssh_key(value: Option<&Value>, key: &SymmetricKey) -> Result<Option<DecryptedSshKey>, Error> {
    let Some(ssh_key) = value else {
        return Ok(None);
    };

    Ok(Some(DecryptedSshKey {
        private_key: decrypt_value_string(ssh_key.get("privateKey"), key)?,
        public_key: decrypt_value_string(ssh_key.get("publicKey"), key)?,
        fingerprint: decrypt_value_string(ssh_key.get("fingerprint"), key)?,
    }))
}

fn decrypt_attachments(
    value: Option<&Value>,
    item_key: &SymmetricKey,
    common: &CommonArgs,
) -> Result<Vec<DecryptedAttachment>, Error> {
    let Some(attachments) = value.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    attachments
        .iter()
        .map(|attachment| {
            let key = match attachment.get("key").and_then(Value::as_str) {
                Some(enc) => Some(SymmetricKey::from_bytes(decrypt_cipher_string_to_bytes(enc, item_key, None)?)?),
                None => {
                    log_verbose(
                        "decrypt",
                        common,
                        format!(
                            "Attachment {} has no explicit key; falling back to cipher key",
                            attachment.get("id").and_then(Value::as_str).unwrap_or_default()
                        ),
                    );
                    Some(item_key.clone())
                }
            };
            Ok(DecryptedAttachment {
                id: get_required_str(attachment, "id")?.to_owned(),
                file_name: decrypt_value_string(attachment.get("fileName"), item_key)?
                    .unwrap_or_else(|| "attachment".to_owned()),
                size: attachment.get("size").and_then(Value::as_str).map(str::to_owned),
                url: get_required_str(attachment, "url")?.to_owned(),
                mime: None,
                key,
            })
        })
        .collect()
}

async fn load_decrypted_export(common: CommonArgs, master_password: Option<String>) -> Result<DecryptedExport, Error> {
    let bundle = login_and_fetch_sync(common.clone(), false).await?;
    let master_password = resolve_master_password(master_password, "Master password: ")?;
    let keys = derive_account_keys(&bundle, &master_password, &common)?;
    decrypt_export(&bundle.sync, &keys, &common)
}

fn compute_local_stats(sync: &Value, export: &DecryptedExport) -> LocalStats {
    let mut ciphers_by_type = HashMap::new();
    let mut attachment_count = 0usize;
    let mut total_attachment_bytes = 0u64;
    let mut items_with_attachments = 0usize;
    let mut organization_owned_count = 0usize;
    let mut personal_owned_count = 0usize;
    let mut trashed_count = 0usize;

    for item in &export.items {
        *ciphers_by_type.entry(cipher_type_name(item.r#type).to_owned()).or_insert(0) += 1;
        if item.organization_id.is_some() {
            organization_owned_count += 1;
        } else {
            personal_owned_count += 1;
        }
        if item.deleted_date.is_some() {
            trashed_count += 1;
        }
        if !item.attachments.is_empty() {
            items_with_attachments += 1;
        }
        attachment_count += item.attachments.len();
        total_attachment_bytes += item
            .attachments
            .iter()
            .filter_map(|attachment| attachment.size.as_deref())
            .filter_map(|size| size.parse::<u64>().ok())
            .sum::<u64>();
    }

    LocalStats {
        total_ciphers: export.items.len(),
        ciphers_by_type,
        folder_count: sync.get("folders").and_then(Value::as_array).map_or(0, Vec::len),
        collection_count: sync.get("collections").and_then(Value::as_array).map_or(0, Vec::len),
        send_count: sync.get("sends").and_then(Value::as_array).map_or(0, Vec::len),
        attachment_count,
        total_attachment_bytes,
        items_with_attachments,
        trashed_count,
        organization_owned_count,
        personal_owned_count,
    }
}

fn search_items<'a>(items: &'a [DecryptedCipher], query: &str, mode: SearchMode) -> Vec<&'a DecryptedCipher> {
    let needle = query.trim().to_lowercase();
    items
        .iter()
        .filter(|item| match mode {
            SearchMode::Domain => item.search_domains.iter().any(|domain| domain.contains(&needle)),
            SearchMode::Name => {
                item.name.to_lowercase().contains(&needle)
                    || item
                        .login
                        .as_ref()
                        .and_then(|login| login.username.as_ref())
                        .is_some_and(|username| username.to_lowercase().contains(&needle))
                    || item.login.as_ref().is_some_and(|login| {
                        login
                            .uris
                            .iter()
                            .any(|uri| uri.uri.as_ref().is_some_and(|value| value.to_lowercase().contains(&needle)))
                    })
            }
            SearchMode::All => {
                item.search_blob.contains(&needle) || item.search_domains.iter().any(|domain| domain.contains(&needle))
            }
        })
        .collect()
}

fn parse_totp_value(value: &str) -> Result<TotpConfig, Error> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::new("Invalid TOTP value", "empty TOTP value"));
    }
    if trimmed.starts_with("otpauth://") {
        return parse_otpauth_totp(trimmed);
    }

    Ok(TotpConfig {
        secret: decode_totp_secret(trimmed)?,
        period: 30,
        digits: 6,
        algorithm: TotpAlgorithm::Sha1,
    })
}

fn parse_otpauth_totp(value: &str) -> Result<TotpConfig, Error> {
    let url = url::Url::parse(value).map_err(|e| Error::new("Invalid TOTP URI", e.to_string()))?;
    if url.scheme() != "otpauth" {
        return Err(Error::new("Invalid TOTP URI", format!("unsupported scheme {}", url.scheme())));
    }
    if url.host_str() != Some("totp") {
        return Err(Error::new("Invalid TOTP URI", format!("unsupported TOTP kind {:?}", url.host_str())));
    }

    let mut secret = None;
    let mut period = 30u64;
    let mut digits = 6u32;
    let mut algorithm = TotpAlgorithm::Sha1;

    for (key, value) in url.query_pairs() {
        match key.as_ref() {
            "secret" => secret = Some(decode_totp_secret(value.as_ref())?),
            "period" => {
                period = value.parse::<u64>().map_err(|e| Error::new("Invalid TOTP URI", e.to_string()))?;
            }
            "digits" => {
                digits = value.parse::<u32>().map_err(|e| Error::new("Invalid TOTP URI", e.to_string()))?;
            }
            "algorithm" => {
                algorithm = match value.to_ascii_uppercase().as_str() {
                    "SHA1" => TotpAlgorithm::Sha1,
                    "SHA256" => TotpAlgorithm::Sha256,
                    "SHA512" => TotpAlgorithm::Sha512,
                    other => TotpAlgorithm::Unsupported(other.to_owned()),
                };
            }
            _ => {}
        }
    }

    let secret = secret.ok_or_else(|| Error::new("Invalid TOTP URI", "missing secret parameter"))?;
    if period == 0 {
        return Err(Error::new("Invalid TOTP URI", "period must be greater than zero"));
    }
    if digits == 0 {
        return Err(Error::new("Invalid TOTP URI", "digits must be greater than zero"));
    }

    Ok(TotpConfig {
        secret,
        period,
        digits,
        algorithm,
    })
}

fn decode_totp_secret(value: &str) -> Result<Vec<u8>, Error> {
    let normalized = value.chars().filter(|ch| !ch.is_ascii_whitespace()).collect::<String>();
    if normalized.is_empty() {
        return Err(Error::new("Invalid TOTP value", "empty TOTP secret"));
    }

    let mut padded = normalized.to_ascii_uppercase();
    let remainder = padded.len() % 8;
    if remainder != 0 {
        padded.extend(std::iter::repeat('=').take(8 - remainder));
    }
    data_encoding::BASE32.decode(padded.as_bytes()).map_err(|e| Error::new("Invalid TOTP value", e.to_string()))
}

fn generate_totp_code(config: &TotpConfig, now: u64) -> Result<TotpCode, Error> {
    let code = match &config.algorithm {
        TotpAlgorithm::Sha1 => totp_custom::<Sha1>(config.period, config.digits, &config.secret, now),
        TotpAlgorithm::Sha256 => totp_custom::<Sha256>(config.period, config.digits, &config.secret, now),
        TotpAlgorithm::Sha512 => totp_custom::<Sha512>(config.period, config.digits, &config.secret, now),
        TotpAlgorithm::Unsupported(name) => {
            return Err(Error::new("unsupported TOTP algorithm", format!("unsupported TOTP algorithm {name}")));
        }
    };

    Ok(TotpCode {
        code,
        seconds_remaining: seconds_remaining_in_period(config.period, now),
    })
}

fn seconds_remaining_in_period(period: u64, now: u64) -> u64 {
    let elapsed = now % period;
    if elapsed == 0 {
        period
    } else {
        period - elapsed
    }
}

fn unix_timestamp_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_else(|_| Duration::from_secs(0)).as_secs()
}

fn derive_master_key(master_password: &str, email: &str, token: &TokenResponse) -> Result<Vec<u8>, Error> {
    let kdf = token.kdf.unwrap_or(0);
    match kdf {
        0 => {
            let iterations = token.kdf_iterations.unwrap_or(600_000);
            let mut out = vec![0u8; PBKDF2_OUTPUT_LEN];
            pbkdf2::derive(
                pbkdf2::PBKDF2_HMAC_SHA256,
                std::num::NonZeroU32::new(iterations).map_res("PBKDF2 iterations must be non-zero")?,
                email.as_bytes(),
                master_password.as_bytes(),
                &mut out,
            );
            Ok(out)
        }
        1 => {
            let memory_mib = token.kdf_memory.unwrap_or(64);
            let iterations = token.kdf_iterations.unwrap_or(3);
            let parallelism = token.kdf_parallelism.unwrap_or(4);
            let mut params = ParamsBuilder::new();
            params.m_cost(memory_mib.saturating_mul(1024));
            params.t_cost(iterations);
            params.p_cost(parallelism);
            let params = params.build().map_err(|e| Error::new("Invalid Argon2 parameters", e.to_string()))?;
            let argon = Argon2::new(ArgonAlgorithm::Argon2id, ArgonVersion::V0x13, params);
            let mut out = vec![0u8; PBKDF2_OUTPUT_LEN];
            let salt_sha256 = {
                let mut ctx = Context::new(&SHA256);
                ctx.update(email.as_bytes());
                let digest = ctx.finish();
                digest.as_ref().to_vec()
            };
            argon
                .hash_password_into(master_password.as_bytes(), &salt_sha256, &mut out)
                .map_err(|e| Error::new("Failed to derive Argon2 master key", e.to_string()))?;
            Ok(out)
        }
        _ => Err(Error::new("Unsupported KDF type", kdf.to_string())),
    }
}

fn stretch_master_key_candidates(master_key: &[u8]) -> Result<Vec<SymmetricKey>, Error> {
    let mut candidates = Vec::new();

    let enc = hkdf_expand_from_prk_32(master_key, b"enc")?;
    let mac = hkdf_expand_from_prk_32(master_key, b"mac")?;
    let mut joined = enc;
    joined.extend_from_slice(&mac);
    candidates.push(SymmetricKey::from_bytes(joined)?);

    // Legacy user keys may still be wrapped directly with AES-CBC using the 32-byte master key.
    candidates.push(SymmetricKey::from_bytes(master_key.to_vec())?);

    Ok(candidates)
}

fn hkdf_expand_from_prk_32(prk: &[u8], info: &[u8]) -> Result<Vec<u8>, Error> {
    struct OutputLen32;
    impl hkdf::KeyType for OutputLen32 {
        fn len(&self) -> usize {
            32
        }
    }

    let prk = hkdf::Prk::new_less_safe(hkdf::HKDF_SHA256, prk);
    let info_parts = [info];
    let okm = prk.expand(&info_parts, OutputLen32).map_err(|_| Error::new("Failed to expand HKDF key", ""))?;
    let mut out = [0u8; 32];
    okm.fill(&mut out).map_err(|_| Error::new("Failed to derive HKDF output", ""))?;
    Ok(out.to_vec())
}

fn decrypt_user_key_with_candidates(
    key_cipher: &str,
    candidates: &[SymmetricKey],
    common: &CommonArgs,
) -> Result<Vec<u8>, Error> {
    for (idx, candidate) in candidates.iter().enumerate() {
        match decrypt_cipher_string_to_bytes(key_cipher, candidate, None) {
            Ok(bytes) => {
                if idx > 0 {
                    log_verbose(
                        "decrypt",
                        common,
                        format!("Recovered user key using fallback master-key stretch variant {}", idx + 1),
                    );
                }
                return Ok(bytes);
            }
            Err(err) => {
                log_verbose("decrypt", common, format!("Master-key stretch variant {} failed: {}", idx + 1, err));
            }
        }

        match decrypt_cipher_string_to_bytes_allow_bad_mac(key_cipher, candidate) {
            Ok(bytes) if looks_like_user_key(&bytes) => {
                log_verbose(
                    "decrypt",
                    common,
                    format!("Recovered user key using fallback master-key stretch variant {} with MAC bypass", idx + 1),
                );
                return Ok(bytes);
            }
            Ok(bytes) => {
                log_verbose(
                    "decrypt",
                    common,
                    format!(
                        "Master-key stretch variant {} produced {} bytes with MAC bypass, but payload was not a valid user key",
                        idx + 1,
                        bytes.len()
                    ),
                );
            }
            Err(err) => {
                log_verbose(
                    "decrypt",
                    common,
                    format!("Master-key stretch variant {} also failed with MAC bypass: {}", idx + 1, err),
                );
            }
        }
    }

    Err(Error::new("Failed to decrypt protected user key", "all master-key stretch variants failed"))
}

fn decrypt_cipher_string_to_bytes_allow_bad_mac(ciphertext: &str, key: &SymmetricKey) -> Result<Vec<u8>, Error> {
    let (enc_type, pieces) = parse_cipher_string(ciphertext)?;
    match enc_type {
        0 => {
            if pieces.len() != 2 {
                return Err(Error::new("Malformed AES-CBC cipher string", ciphertext));
            }
            let iv = decode_base64_piece(pieces[0])?;
            let data = decode_base64_piece(pieces[1])?;
            decrypt_aes_cbc(&key.enc, &iv, &data)
        }
        2 => {
            if pieces.len() != 3 {
                return Err(Error::new("Malformed AES-CBC-HMAC cipher string", ciphertext));
            }
            let iv = decode_base64_piece(pieces[0])?;
            let data = decode_base64_piece(pieces[1])?;
            decrypt_aes_cbc(&key.enc, &iv, &data)
        }
        other => Err(Error::new("Unsupported encryption type for MAC-bypass decryption", other.to_string())),
    }
}

fn looks_like_user_key(bytes: &[u8]) -> bool {
    matches!(bytes.len(), 32 | 64)
}

impl SymmetricKey {
    fn from_bytes(bytes: Vec<u8>) -> Result<Self, Error> {
        match bytes.len() {
            32 => Ok(Self {
                enc: bytes,
                mac: None,
            }),
            64 => Ok(Self {
                enc: bytes[..32].to_vec(),
                mac: Some(bytes[32..].to_vec()),
            }),
            len => Err(Error::new("Unsupported symmetric key length", len.to_string())),
        }
    }
}

fn decrypt_optional_string_field(value: &Value, field: &str, key: &SymmetricKey) -> Result<Option<String>, Error> {
    decrypt_value_string(value.get(field), key)
}

fn decrypt_value_string(value: Option<&Value>, key: &SymmetricKey) -> Result<Option<String>, Error> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let ciphertext = value.as_str().ok_or_else(|| Error::new("Encrypted field must be a string", value.to_string()))?;
    decrypt_cipher_string_to_string(ciphertext, key, None).map(Some)
}

fn decrypt_cipher_string_to_string(
    ciphertext: &str,
    key: &SymmetricKey,
    rsa_key: Option<&Rsa<openssl::pkey::Private>>,
) -> Result<String, Error> {
    let bytes = decrypt_cipher_string_to_bytes(ciphertext, key, rsa_key)?;
    String::from_utf8(bytes).map_err(|e| Error::new("Decrypted value is not valid UTF-8", e.to_string()))
}

fn decrypt_cipher_string_to_bytes(
    ciphertext: &str,
    key: &SymmetricKey,
    rsa_key: Option<&Rsa<openssl::pkey::Private>>,
) -> Result<Vec<u8>, Error> {
    let (enc_type, pieces) = parse_cipher_string(ciphertext)?;
    match enc_type {
        0 => {
            if pieces.len() != 2 {
                return Err(Error::new("Malformed AES-CBC cipher string", ciphertext));
            }
            let iv = decode_base64_piece(pieces[0])?;
            let data = decode_base64_piece(pieces[1])?;
            decrypt_aes_cbc(&key.enc, &iv, &data)
        }
        2 => {
            if pieces.len() != 3 {
                return Err(Error::new("Malformed AES-CBC-HMAC cipher string", ciphertext));
            }
            let iv = decode_base64_piece(pieces[0])?;
            let data = decode_base64_piece(pieces[1])?;
            let mac = decode_base64_piece(pieces[2])?;
            verify_mac(key, &iv, &data, &mac)?;
            decrypt_aes_cbc(&key.enc, &iv, &data)
        }
        3 | 4 => {
            let Some(rsa_key) = rsa_key else {
                return Err(Error::new("Missing RSA key for asymmetric decryption", enc_type.to_string()));
            };
            if pieces.len() != 1 {
                return Err(Error::new("Malformed RSA cipher string", ciphertext));
            }
            let data = decode_base64_piece(pieces[0])?;
            decrypt_rsa(rsa_key, &data, enc_type == 3)
        }
        other => Err(Error::new("Unsupported encryption type", other.to_string())),
    }
}

fn decrypt_rsa_cipher_string_to_bytes(
    ciphertext: &str,
    rsa_key: &Rsa<openssl::pkey::Private>,
) -> Result<Vec<u8>, Error> {
    let (enc_type, pieces) = parse_cipher_string(ciphertext)?;
    match enc_type {
        3 | 4 => {
            let piece = pieces.first().copied().map_res("Malformed RSA cipher string")?;
            let data = decode_base64_piece(piece)?;
            decrypt_rsa(rsa_key, &data, enc_type == 3)
        }
        other => Err(Error::new("Unsupported RSA encryption type", other.to_string())),
    }
}

fn parse_decrypted_private_key(private_key_bytes: &[u8]) -> Result<Rsa<openssl::pkey::Private>, Error> {
    if let Ok(rsa) = Rsa::private_key_from_der(private_key_bytes) {
        return Ok(rsa);
    }

    if let Ok(pkey) = PKey::private_key_from_der(private_key_bytes) {
        return pkey.rsa().map_err(|e| Error::new("Failed to extract RSA private key from PKCS#8 DER", e.to_string()));
    }

    if let Ok(rsa) = Rsa::private_key_from_pem(private_key_bytes) {
        return Ok(rsa);
    }

    if let Ok(pkey) = PKey::private_key_from_pem(private_key_bytes) {
        return pkey.rsa().map_err(|e| Error::new("Failed to extract RSA private key from PKCS#8 PEM", e.to_string()));
    }

    Err(Error::new("Failed to parse decrypted private key", "expected PKCS#8 DER/PEM or PKCS#1 DER/PEM"))
}

fn decrypt_bytes_with_key(ciphertext: &[u8], key: &SymmetricKey) -> Result<Vec<u8>, Error> {
    if let Ok(text) = std::str::from_utf8(ciphertext) {
        if text.contains('.') && text.contains('|') {
            return decrypt_cipher_string_to_bytes(text.trim(), key, None);
        }
    }

    decrypt_buffer_enc_string_to_bytes(ciphertext, key)
}

fn decrypt_buffer_enc_string_to_bytes(ciphertext: &[u8], key: &SymmetricKey) -> Result<Vec<u8>, Error> {
    let Some((&enc_type, rest)) = ciphertext.split_first() else {
        return Err(Error::new("Attachment data is not a valid EncString", "empty buffer"));
    };

    match enc_type {
        0 => {
            if rest.len() < 16 {
                return Err(Error::new("Malformed attachment AES-CBC buffer", rest.len().to_string()));
            }
            let (iv, data) = rest.split_at(16);
            decrypt_aes_cbc(&key.enc, iv, data)
        }
        2 => {
            if rest.len() < 48 {
                return Err(Error::new("Malformed attachment AES-CBC-HMAC buffer", rest.len().to_string()));
            }
            let (iv, rest) = rest.split_at(16);
            let (mac, data) = rest.split_at(32);
            verify_mac(key, iv, data, mac)?;
            decrypt_aes_cbc(&key.enc, iv, data)
        }
        7 => Err(Error::new(
            "Unsupported attachment encryption type",
            "COSE/XChaCha20 attachments are not yet implemented",
        )),
        other => Err(Error::new("Unsupported attachment encryption type", other.to_string())),
    }
}

fn parse_cipher_string(ciphertext: &str) -> Result<(u32, Vec<&str>), Error> {
    let mut parts = ciphertext.splitn(2, '.');
    let enc_type = parts
        .next()
        .map_res("Malformed cipher string")?
        .parse::<u32>()
        .map_err(|e| Error::new("Invalid cipher string encryption type", e.to_string()))?;
    let pieces = parts.next().map_res("Malformed cipher string")?.split('|').collect::<Vec<_>>();
    Ok((enc_type, pieces))
}

fn decode_base64_piece(input: &str) -> Result<Vec<u8>, Error> {
    BASE64.decode(input.trim().as_bytes()).map_err(|e| Error::new("Invalid base64 payload", e.to_string()))
}

fn verify_mac(key: &SymmetricKey, iv: &[u8], data: &[u8], mac: &[u8]) -> Result<(), Error> {
    let mac_key =
        key.mac.as_ref().map_res("Cipher requires an HMAC key but the selected symmetric key does not have one")?;
    let signing_key = hmac::Key::new(hmac::HMAC_SHA256, mac_key);
    let mut input = Vec::with_capacity(iv.len() + data.len());
    input.extend_from_slice(iv);
    input.extend_from_slice(data);
    hmac::verify(&signing_key, &input, mac).map_err(|_| Error::new("Cipher MAC validation failed", "hmac mismatch"))
}

fn decrypt_aes_cbc(key: &[u8], iv: &[u8], data: &[u8]) -> Result<Vec<u8>, Error> {
    decrypt(Cipher::aes_256_cbc(), key, Some(iv), data).map_res("AES-CBC decryption failed")
}

fn decrypt_rsa(rsa_key: &Rsa<openssl::pkey::Private>, data: &[u8], sha256: bool) -> Result<Vec<u8>, Error> {
    let mut buf = vec![0u8; rsa_key.size() as usize];
    if sha256 {
        let pkey = PKey::from_rsa(rsa_key.clone()).map_res("Failed to prepare RSA key")?;
        let mut decryptor = openssl::encrypt::Decrypter::new(&pkey).map_res("Failed to create RSA decryptor")?;
        decryptor.set_rsa_padding(Padding::PKCS1_OAEP).map_res("Failed to configure RSA padding")?;
        decryptor
            .set_rsa_oaep_md(openssl::hash::MessageDigest::sha256())
            .map_res("Failed to configure RSA OAEP digest")?;
        decryptor
            .set_rsa_mgf1_md(openssl::hash::MessageDigest::sha256())
            .map_res("Failed to configure RSA MGF1 digest")?;
        let len = decryptor.decrypt(data, &mut buf).map_res("RSA-OAEP-SHA256 decryption failed")?;
        buf.truncate(len);
        Ok(buf)
    } else {
        let len =
            rsa_key.private_decrypt(data, &mut buf, Padding::PKCS1_OAEP).map_res("RSA-OAEP-SHA1 decryption failed")?;
        buf.truncate(len);
        Ok(buf)
    }
}

async fn ensure_success_response(
    response: reqwest::Response,
    context: &str,
    tag: &str,
    common: &CommonArgs,
) -> Result<reqwest::Response, Error> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.unwrap_or_else(|err| format!("Unable to read error response body: {err}"));
    log_verbose(tag, common, format!("{context}: status={status}, body={}", body.trim()));
    Err(Error::new(format!("{context} ({status})"), body))
}

fn resolve_master_password(master_password: Option<String>, prompt: &str) -> Result<String, Error> {
    match master_password {
        Some(password) => Ok(password),
        None => rpassword::prompt_password(prompt).map_res("Failed to read master password"),
    }
}

fn get_required_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, Error> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new(format!("Missing required key {key}"), value.to_string()))
}

fn normalize_base_url(url: String) -> String {
    url.trim_end_matches('/').to_owned()
}

fn normalize_client_id(client_id: String) -> String {
    if client_id.starts_with("user.") {
        client_id
    } else {
        format!("user.{client_id}")
    }
}

fn normalize_email(email: &str) -> String {
    email.trim().to_lowercase()
}

fn normalize_uri_host(value: &str) -> Option<String> {
    let parsed = url::Url::parse(value).or_else(|_| url::Url::parse(&format!("https://{value}"))).ok()?;
    parsed.host_str().map(|host| host.to_lowercase())
}

fn cipher_type_name(cipher_type: i64) -> &'static str {
    match cipher_type {
        1 => "login",
        2 => "secureNote",
        3 => "card",
        4 => "identity",
        5 => "sshKey",
        _ => "unknown",
    }
}

fn sanitize_file_name(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => ch,
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "attachment".to_owned()
    } else {
        sanitized
    }
}

fn default_fetch_dir() -> PathBuf {
    let ts = chrono::Utc::now().format("%Y%m%d%H%M%S");
    std::env::temp_dir().join(format!("vaultwarden-fetch-{ts}"))
}

fn path_to_forward_slashes(path: &Path) -> String {
    path.components().map(|component| component.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/")
}

fn format_os_args(args: &[OsString]) -> String {
    args.iter().map(|arg| arg.to_string_lossy()).collect::<Vec<_>>().join(" ")
}

fn log_info(tag: &str, _common: &CommonArgs, message: String) {
    eprintln!("[{tag}] {message}");
}

fn log_verbose(tag: &str, common: &CommonArgs, message: String) {
    if common.verbose {
        eprintln!("[{tag}:verbose] {message}");
    }
}

pub fn print_help_and_exit() -> ! {
    let version = VERSION.unwrap_or("(Version info from Git not present)");
    println!("Vaultwarden {version}");
    print!("{}", crate::help_text());
    exit(0);
}

pub fn print_version_and_exit() -> ! {
    crate::config::SKIP_CONFIG_VALIDATION.store(true, std::sync::atomic::Ordering::Relaxed);
    let web_vault_version = crate::util::get_web_vault_version();
    let version = VERSION.unwrap_or("(Version info from Git not present)");
    println!("Vaultwarden {version}");
    println!("Web-Vault {web_vault_version}");
    exit(0);
}

#[cfg(test)]
mod tests {
    use super::{
        collect_attachment_downloads, decrypt_aes_cbc, decrypt_bytes_with_key, decrypt_cipher_string_to_bytes,
        generate_totp_code, hkdf_expand_from_prk_32, normalize_base_url, normalize_client_id, normalize_uri_host,
        parse_cipher_string, parse_totp_value, path_to_forward_slashes, sanitize_file_name,
        seconds_remaining_in_period, SymmetricKey, TotpAlgorithm,
    };
    use data_encoding::BASE64;
    use openssl::symm::{encrypt, Cipher};
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn normalize_base_url_removes_trailing_slashes() {
        assert_eq!(normalize_base_url("https://example.com///".to_owned()), "https://example.com");
    }

    #[test]
    fn normalize_client_id_adds_user_prefix() {
        assert_eq!(normalize_client_id("1234".to_owned()), "user.1234");
        assert_eq!(normalize_client_id("user.1234".to_owned()), "user.1234");
    }

    #[test]
    fn collect_attachment_downloads_reads_cipher_attachments() {
        let sync_json = json!({
            "ciphers": [
                {
                    "id": "cipher-1",
                    "attachments": [
                        {
                            "id": "att-1",
                            "url": "https://vault/attachments/cipher-1/att-1?token=x",
                            "fileName": "enc-name",
                            "size": "10"
                        }
                    ]
                }
            ]
        });

        let attachments = collect_attachment_downloads(&sync_json);
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].cipher_id, "cipher-1");
        assert_eq!(attachments[0].relative_path, Path::new("attachments").join("cipher-1").join("att-1"));
    }

    #[test]
    fn path_to_forward_slashes_normalizes_paths() {
        let path = Path::new("attachments").join("cipher-1").join("att-1");
        assert_eq!(path_to_forward_slashes(&path), "attachments/cipher-1/att-1");
    }

    #[test]
    fn sanitize_file_name_replaces_path_unsafe_chars() {
        assert_eq!(sanitize_file_name("a/b:c"), "a_b_c");
    }

    #[test]
    fn normalize_uri_host_extracts_host() {
        assert_eq!(normalize_uri_host("https://Example.com/login").as_deref(), Some("example.com"));
        assert_eq!(normalize_uri_host("example.com/path").as_deref(), Some("example.com"));
    }

    #[test]
    fn parse_cipher_string_reads_type_and_pieces() {
        let (enc_type, pieces) = parse_cipher_string("2.aaa|bbb|ccc").unwrap();
        assert_eq!(enc_type, 2);
        assert_eq!(pieces, vec!["aaa", "bbb", "ccc"]);
    }

    #[test]
    fn decrypt_cipher_string_decrypts_aes_cbc_hmac() {
        let key = SymmetricKey::from_bytes((0..64).map(|i| i as u8).collect()).unwrap();
        let iv = [7u8; 16];
        let plaintext = br#"hello vaultwarden"#;
        let ciphertext = encrypt(Cipher::aes_256_cbc(), &key.enc, Some(&iv), plaintext).unwrap();
        let mut mac_input = Vec::new();
        mac_input.extend_from_slice(&iv);
        mac_input.extend_from_slice(&ciphertext);
        let mac =
            ring::hmac::sign(&ring::hmac::Key::new(ring::hmac::HMAC_SHA256, key.mac.as_ref().unwrap()), &mac_input);
        let enc_string =
            format!("2.{}|{}|{}", BASE64.encode(&iv), BASE64.encode(&ciphertext), BASE64.encode(mac.as_ref()));

        let decrypted = decrypt_cipher_string_to_bytes(&enc_string, &key, None).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn decrypt_aes_cbc_roundtrip() {
        let key = [1u8; 32];
        let iv = [2u8; 16];
        let plaintext = b"bitwarden-cli";
        let ciphertext = encrypt(Cipher::aes_256_cbc(), &key, Some(&iv), plaintext).unwrap();
        let decrypted = decrypt_aes_cbc(&key, &iv, &ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn hkdf_expand_from_prk_matches_bitwarden_sdk_stretch_key_vectors() {
        let master_key = [
            31, 79, 104, 226, 150, 71, 177, 90, 194, 80, 172, 209, 17, 129, 132, 81, 138, 167, 69, 167, 254, 149, 2,
            27, 39, 197, 64, 42, 22, 195, 86, 75,
        ];
        let enc = hkdf_expand_from_prk_32(&master_key, b"enc").unwrap();
        let mac = hkdf_expand_from_prk_32(&master_key, b"mac").unwrap();

        assert_eq!(
            enc,
            vec![
                111, 31, 178, 45, 238, 152, 37, 114, 143, 215, 124, 83, 135, 173, 195, 23, 142, 134, 120, 249, 61, 132,
                163, 182, 113, 197, 189, 204, 188, 21, 237, 96
            ]
        );
        assert_eq!(
            mac,
            vec![
                221, 127, 206, 234, 101, 27, 202, 38, 86, 52, 34, 28, 78, 28, 185, 16, 48, 61, 127, 166, 209, 247, 194,
                87, 232, 26, 48, 85, 193, 249, 179, 155
            ]
        );
    }

    #[test]
    fn decrypt_bytes_with_key_decrypts_binary_attachment_buffer() {
        let key = SymmetricKey::from_bytes((0..64).map(|i| i as u8).collect()).unwrap();
        let iv = [7u8; 16];
        let plaintext = br#"hello attachment"#;
        let ciphertext = encrypt(Cipher::aes_256_cbc(), &key.enc, Some(&iv), plaintext).unwrap();
        let mut mac_input = Vec::new();
        mac_input.extend_from_slice(&iv);
        mac_input.extend_from_slice(&ciphertext);
        let mac =
            ring::hmac::sign(&ring::hmac::Key::new(ring::hmac::HMAC_SHA256, key.mac.as_ref().unwrap()), &mac_input);

        let mut attachment_buffer = Vec::new();
        attachment_buffer.push(2);
        attachment_buffer.extend_from_slice(&iv);
        attachment_buffer.extend_from_slice(mac.as_ref());
        attachment_buffer.extend_from_slice(&ciphertext);

        let decrypted = decrypt_bytes_with_key(&attachment_buffer, &key).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn parse_totp_value_accepts_raw_base32_secret() {
        let config = parse_totp_value("JBSWY3DPEHPK3PXP").unwrap();
        assert_eq!(config.secret, b"Hello!\xde\xad\xbe\xef".to_vec());
        assert_eq!(config.period, 30);
        assert_eq!(config.digits, 6);
        assert!(matches!(config.algorithm, TotpAlgorithm::Sha1));
    }

    #[test]
    fn parse_totp_value_accepts_default_otpauth_uri() {
        let config = parse_totp_value("otpauth://totp/Test?secret=JBSWY3DPEHPK3PXP").unwrap();
        assert_eq!(config.secret, b"Hello!\xde\xad\xbe\xef".to_vec());
        assert_eq!(config.period, 30);
        assert_eq!(config.digits, 6);
        assert!(matches!(config.algorithm, TotpAlgorithm::Sha1));
    }

    #[test]
    fn parse_totp_value_reads_explicit_digits_period_and_algorithm() {
        let config =
            parse_totp_value("otpauth://totp/Test?secret=JBSWY3DPEHPK3PXP&period=45&digits=8&algorithm=SHA256")
                .unwrap();
        assert_eq!(config.period, 45);
        assert_eq!(config.digits, 8);
        assert!(matches!(config.algorithm, TotpAlgorithm::Sha256));
    }

    #[test]
    fn parse_totp_value_rejects_missing_secret() {
        let err = parse_totp_value("otpauth://totp/Test?period=45").unwrap_err();
        assert!(format!("{err:?}").contains("missing secret"));
    }

    #[test]
    fn parse_totp_value_marks_unsupported_algorithm() {
        let config = parse_totp_value("otpauth://totp/Test?secret=JBSWY3DPEHPK3PXP&algorithm=MD5").unwrap();
        assert!(matches!(config.algorithm, TotpAlgorithm::Unsupported(ref name) if name == "MD5"));
    }

    #[test]
    fn generate_totp_code_matches_known_sha1_vector() {
        let config = parse_totp_value("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ").unwrap();
        let code = generate_totp_code(&config, 59).unwrap();
        assert_eq!(code.code, "287082");
        assert_eq!(code.seconds_remaining, 1);
    }

    #[test]
    fn generate_totp_code_honors_custom_digits_and_period() {
        let config = parse_totp_value(
            "otpauth://totp/Test?secret=GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ&period=60&digits=8&algorithm=SHA1",
        )
        .unwrap();
        let code = generate_totp_code(&config, 59).unwrap();
        assert_eq!(code.code, "84755224");
        assert_eq!(code.seconds_remaining, 1);
    }

    #[test]
    fn generate_totp_code_rejects_unsupported_algorithm() {
        let config = parse_totp_value("otpauth://totp/Test?secret=JBSWY3DPEHPK3PXP&algorithm=MD5").unwrap();
        let err = generate_totp_code(&config, 59).unwrap_err();
        assert!(format!("{err:?}").contains("unsupported TOTP algorithm"));
    }

    #[test]
    fn seconds_remaining_handles_step_boundaries() {
        assert_eq!(seconds_remaining_in_period(30, 0), 30);
        assert_eq!(seconds_remaining_in_period(30, 29), 1);
        assert_eq!(seconds_remaining_in_period(30, 30), 30);
        assert_eq!(seconds_remaining_in_period(30, 31), 29);
    }
}
