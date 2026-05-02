use std::{
    collections::HashMap,
    ffi::OsString,
    io::{self, Stdout, Write},
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
    symm::{decrypt, encrypt, Cipher},
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
use reqwest::multipart::{Form, Part};
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

use crate::{crypto, Error, MapResult, VERSION};

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
    upload --server <url> --client-id <user.uuid> --client-secret <secret> --input <dump-dir>
                                       Upload a decrypted dump into another account
                                       Flags: [--diagnostics] [--skip-attachments]
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
struct UploadArgs {
    common: CommonArgs,
    input_dir: PathBuf,
    master_password: Option<String>,
    diagnostics: bool,
    skip_attachments: bool,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecryptedFolder {
    id: String,
    name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecryptedField {
    name: Option<String>,
    value: Option<String>,
    #[serde(rename = "type")]
    field_type: Option<i64>,
    linked_id: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecryptedPasswordHistory {
    password: Option<String>,
    last_used_date: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecryptedUri {
    uri: Option<String>,
    match_type: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecryptedAttachment {
    id: String,
    file_name: String,
    size: Option<String>,
    url: String,
    mime: Option<String>,
    #[serde(default, skip)]
    key: Option<SymmetricKey>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecryptedLogin {
    username: Option<String>,
    password: Option<String>,
    totp: Option<String>,
    uris: Vec<DecryptedUri>,
    password_history: Vec<DecryptedPasswordHistory>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecryptedCard {
    cardholder_name: Option<String>,
    brand: Option<String>,
    number: Option<String>,
    exp_month: Option<String>,
    exp_year: Option<String>,
    code: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecryptedSecureNote {
    note_type: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecryptedSshKey {
    private_key: Option<String>,
    public_key: Option<String>,
    fingerprint: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    #[serde(default, skip)]
    search_blob: String,
    #[serde(default, skip)]
    search_domains: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Debug)]
struct UploadSource {
    export: DecryptedExport,
    attachment_index: HashMap<(String, String), DecryptedAttachmentIndexEntry>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DuplicateDecision {
    Allow,
    KeepExisting,
    KeepIncoming,
}

impl DuplicateDecision {
    fn label(self) -> &'static str {
        match self {
            DuplicateDecision::Allow => "allow",
            DuplicateDecision::KeepExisting => "keep-existing",
            DuplicateDecision::KeepIncoming => "keep-incoming",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct DiagnosticsEntry {
    source_name: String,
    source_id: String,
    source_json: Value,
    existing_id: Option<String>,
    existing_name: Option<String>,
    existing_json: Option<Value>,
    decision: String,
    action: String,
}

#[derive(Clone, Debug, Serialize)]
struct DiagnosticsReport {
    total_items: usize,
    created: usize,
    skipped: usize,
    updated: usize,
    trashed: usize,
    conflicts: Vec<DiagnosticsEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UploadMode {
    Skip,
    Create,
    Replace,
}

#[derive(Clone, Debug)]
struct UploadPlanItem {
    source: DecryptedCipher,
    target_folder_id: Option<String>,
    mode: UploadMode,
    target_cipher: Option<DecryptedCipher>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DuplicatePickerFocus {
    Source,
    Target,
    Details,
}

#[derive(Clone, Debug)]
struct DuplicatePickerState {
    source_selected: usize,
    target_selected: usize,
    focus: DuplicatePickerFocus,
}

#[derive(Clone, Debug, Serialize)]
struct ValidationCipher {
    r#type: i64,
    name: String,
    notes: Option<String>,
    favorite: Option<bool>,
    folder_id: Option<String>,
    deleted: bool,
    fields: Vec<CanonicalUploadField>,
    login: Option<CanonicalUploadLogin>,
    card: Option<CanonicalUploadCard>,
    identity: Option<CanonicalUploadIdentity>,
    secure_note: Option<CanonicalUploadSecureNote>,
    ssh_key: Option<CanonicalUploadSshKey>,
    attachments: Vec<CanonicalUploadAttachment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttachmentRestoreMode {
    Enforce,
    SkipValidation,
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
        "upload" => {
            let args = parse_upload_args(pargs)?;
            ensure_consumed(pargs)?;
            run_upload(args).await?;
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

fn parse_upload_args(pargs: &mut pico_args::Arguments) -> Result<UploadArgs, Error> {
    let common = parse_common_args(pargs)?;
    let input_dir = pargs
        .opt_value_from_str::<_, PathBuf>("--input")
        .map_err(|e| Error::new("Invalid --input argument", e.to_string()))?
        .map_res("Missing required argument --input")?;

    Ok(UploadArgs {
        common,
        input_dir,
        master_password: parse_master_password_arg(pargs)?,
        diagnostics: pargs.contains("--diagnostics"),
        skip_attachments: pargs.contains("--skip-attachments"),
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

async fn run_upload(args: UploadArgs) -> Result<(), Error> {
    log_info("upload", &args.common, format!("Starting vault upload from {}", args.input_dir.display()));

    let source = load_upload_source(&args.input_dir).await?;
    let bundle = login_and_fetch_sync(args.common.clone(), false).await?;
    let master_password = resolve_master_password(args.master_password, "Master password: ")?;
    let keys = derive_account_keys(&bundle, &master_password, &args.common)?;
    let target_export = decrypt_export_lossy(&bundle.sync, &keys, &args.common);

    let target_folder_map =
        ensure_target_folders(&bundle.session, &source.export.folders, &target_export.folders).await?;
    let source_folder_names = folder_name_map(&source.export.folders);
    let mut existing_signatures = build_existing_signatures(&target_export, &target_export.folders);

    log_info("upload", &args.common, format!(
        "Loaded {} items and {} folders from source dump",
        source.export.items.len(),
        source.export.folders.len()
    ));

    for item in &source.export.items {
        if item.organization_id.is_some() {
            return Err(Error::new(
                "Organization-owned items are not supported by upload",
                format!("item={} id={}", item.name, item.id),
            ));
        }
    }

    let source_items = resolve_source_duplicates(&source.export.items, &source_folder_names, &args.common)?;
    let mut upload_plan = Vec::with_capacity(source_items.len());
    for item in source_items {
        let source_folder_name =
            item.folder_id.as_ref().and_then(|folder_id| source_folder_names.get(folder_id)).cloned();
        let target_folder_id =
            source_folder_name.as_ref().and_then(|folder_name| target_folder_map.get(folder_name)).cloned();
        let source_signature = build_duplicate_signature(&item, source_folder_name.as_deref());
        let target_matches = existing_signatures.get(&source_signature).cloned().unwrap_or_default();
        let (mode, target_cipher) = if target_matches.is_empty() {
            (UploadMode::Create, None)
        } else {
            resolve_target_duplicate(&item, &target_matches, &args.common)?
        };
        upload_plan.push(UploadPlanItem {
            source: item,
            target_folder_id,
            mode,
            target_cipher,
        });
    }

    let mut created = 0usize;
    let mut skipped = 0usize;
    let mut updated = 0usize;
    let mut trashed = 0usize;
    let mut diagnostics_entries = Vec::new();
    let mut skip_attachment_uploads = args.skip_attachments;

    for plan_item in upload_plan {
        let upload_item = prepare_upload_item(&plan_item.source, plan_item.target_folder_id.clone(), &keys.user_key);
        verify_prepared_upload_item(&upload_item)?;
        let existing_opt = plan_item.target_cipher.clone();
        let decision = match plan_item.mode {
            UploadMode::Skip => DuplicateDecision::KeepExisting,
            UploadMode::Create => DuplicateDecision::Allow,
            UploadMode::Replace => DuplicateDecision::KeepIncoming,
        };

        let diag_existing = existing_opt
            .as_ref()
            .map(|ex| (ex.id.clone(), ex.name.clone(), serde_json::to_value(ex).unwrap_or_default()));

        if args.diagnostics {
            let action = match decision {
                DuplicateDecision::Allow => "created",
                DuplicateDecision::KeepExisting => "kept-existing",
                DuplicateDecision::KeepIncoming => "updated",
            };
            let source_owned = upload_item.source.clone();
            diagnostics_entries.push(DiagnosticsEntry {
                source_name: source_owned.name.clone(),
                source_id: source_owned.id.clone(),
                source_json: serde_json::to_value(&source_owned).unwrap_or_default(),
                existing_id: diag_existing.as_ref().map(|(id, _, _)| id.clone()),
                existing_name: diag_existing.as_ref().map(|(_, name, _)| name.clone()),
                existing_json: diag_existing.map(|(_, _, json)| json),
                decision: decision.label().to_string(),
                action: action.to_string(),
            });
        }

        match (plan_item.mode, existing_opt.as_ref()) {
            (UploadMode::Skip, _) => {
                skipped += 1;
                log_info("upload", &args.common, format!("Skipped {}", upload_item.name));
            }
            (UploadMode::Create, _) => {
                let cipher_id = upload_cipher(&bundle.session, &upload_item).await?;
                let mut attachment_mode = if skip_attachment_uploads {
                    AttachmentRestoreMode::SkipValidation
                } else {
                    AttachmentRestoreMode::Enforce
                };
                if !skip_attachment_uploads {
                    match upload_cipher_attachments(
                        &bundle.session,
                        &cipher_id,
                        &upload_item,
                        &source.attachment_index,
                        &args.input_dir,
                    )
                    .await
                    {
                        Ok(()) => {}
                        Err(err) => {
                            if prompt_skip_attachments_on_error(&upload_item.name, &err)? {
                                skip_attachment_uploads = true;
                                attachment_mode = AttachmentRestoreMode::SkipValidation;
                                log_info(
                                    "upload",
                                    &args.common,
                                    format!("Skipping attachment uploads after failure on {}", upload_item.name),
                                );
                            } else {
                                return Err(err);
                            }
                        }
                    }
                }

                if upload_item.deleted_date.is_some() {
                    soft_delete_cipher(&bundle.session, &cipher_id).await?;
                    trashed += 1;
                }

                let verified = verify_uploaded_cipher_with_attachment_prompt(
                    &bundle.session,
                    &keys,
                    &upload_item,
                    &cipher_id,
                    &mut attachment_mode,
                )
                .await?;
                insert_existing_signature(&mut existing_signatures, verified);
                created += 1;
                log_info("upload", &args.common, format!("Created {} (id={})", upload_item.name, cipher_id));
            }
            (UploadMode::Replace, Some(existing_item)) => {
                let mut attachment_mode = if skip_attachment_uploads {
                    AttachmentRestoreMode::SkipValidation
                } else {
                    AttachmentRestoreMode::Enforce
                };
                if let Err(err) = update_cipher(&bundle.session, existing_item, &upload_item).await {
                    if error_indicates_missing_cipher(&err) {
                        log_info(
                            "upload",
                            &args.common,
                            format!(
                                "Target cipher {} disappeared while updating {}; creating a new item instead",
                                existing_item.id, upload_item.name
                            ),
                        );
                        let cipher_id = upload_cipher(&bundle.session, &upload_item).await?;
                        if !skip_attachment_uploads {
                            match upload_cipher_attachments(
                                &bundle.session,
                                &cipher_id,
                                &upload_item,
                                &source.attachment_index,
                                &args.input_dir,
                            )
                            .await
                            {
                                Ok(()) => {}
                            Err(err) => {
                                if prompt_skip_attachments_on_error(&upload_item.name, &err)? {
                                    skip_attachment_uploads = true;
                                    attachment_mode = AttachmentRestoreMode::SkipValidation;
                                    log_info(
                                        "upload",
                                        &args.common,
                                        format!("Skipping attachment uploads after failure on {}", upload_item.name),
                                    );
                                    } else {
                                        return Err(err);
                                    }
                                }
                            }
                        }
                        if upload_item.deleted_date.is_some() {
                            soft_delete_cipher(&bundle.session, &cipher_id).await?;
                            trashed += 1;
                        }
                        let verified = verify_uploaded_cipher_with_attachment_prompt(
                            &bundle.session,
                            &keys,
                            &upload_item,
                            &cipher_id,
                            &mut attachment_mode,
                        )
                        .await?;
                        replace_existing_signature(&mut existing_signatures, &existing_item.id, verified);
                        created += 1;
                        log_info("upload", &args.common, format!("Created {} (id={})", upload_item.name, cipher_id));
                        continue;
                    }
                    return Err(err);
                }
                if !skip_attachment_uploads {
                    match reconcile_cipher_attachments(
                        &bundle.session,
                        &upload_item,
                        existing_item,
                        &source.attachment_index,
                        &args.input_dir,
                    )
                    .await
                    {
                        Ok(()) => {}
                        Err(err) => {
                            if prompt_skip_attachments_on_error(&upload_item.name, &err)? {
                                skip_attachment_uploads = true;
                                attachment_mode = AttachmentRestoreMode::SkipValidation;
                                log_info(
                                    "upload",
                                    &args.common,
                                    format!("Skipping attachment uploads after failure on {}", upload_item.name),
                                );
                            } else {
                                return Err(err);
                            }
                        }
                    }
                }
                if upload_item.deleted_date.is_some() {
                    soft_delete_cipher(&bundle.session, &existing_item.id).await?;
                    trashed += 1;
                } else if existing_item.deleted_date.is_some() {
                    restore_cipher(&bundle.session, &existing_item.id).await?;
                }
                let verified = verify_uploaded_cipher_with_attachment_prompt(
                    &bundle.session,
                    &keys,
                    &upload_item,
                    &existing_item.id,
                    &mut attachment_mode,
                )
                .await?;
                replace_existing_signature(&mut existing_signatures, &existing_item.id, verified);
                updated += 1;
                log_info("upload", &args.common, format!("Updated {} -> {}", upload_item.name, existing_item.id));
            }
            (UploadMode::Replace, None) => {
                skipped += 1;
                log_verbose(
                    "upload",
                    &args.common,
                    format!("Replace mode chosen for {} but no target cipher exists", upload_item.name),
                );
            }
        }
    }

    if args.diagnostics {
        let report = DiagnosticsReport {
            total_items: source.export.items.len(),
            created,
            skipped,
            updated,
            trashed,
            conflicts: diagnostics_entries,
        };
        let report_json =
            serde_json::to_string_pretty(&report).map_res("Failed to format diagnostics report")?;
        eprintln!("\n[upload] diagnostics report:\n{report_json}");
    }

    log_info(
        "upload",
        &args.common,
        format!("Completed upload: created={created} updated={updated} skipped={skipped} trashed={trashed}"),
    );
    Ok(())
}

fn error_indicates_missing_cipher(err: &Error) -> bool {
    let text = format!("{err}");
    text.contains("Cipher doesn't exist")
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalUploadCipher {
    #[serde(rename = "type")]
    r#type: i64,
    name: String,
    notes: Option<String>,
    favorite: Option<bool>,
    folder: Option<String>,
    deleted: bool,
    fields: Vec<CanonicalUploadField>,
    login: Option<CanonicalUploadLogin>,
    card: Option<CanonicalUploadCard>,
    identity: Option<CanonicalUploadIdentity>,
    secure_note: Option<CanonicalUploadSecureNote>,
    ssh_key: Option<CanonicalUploadSshKey>,
    attachments: Vec<CanonicalUploadAttachment>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalUploadField {
    name: Option<String>,
    value: Option<String>,
    #[serde(rename = "type")]
    field_type: Option<i64>,
    linked_id: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalUploadLogin {
    username: Option<String>,
    password: Option<String>,
    totp: Option<String>,
    uris: Vec<CanonicalUploadUri>,
    password_history: Vec<CanonicalUploadPasswordHistory>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalUploadUri {
    uri: Option<String>,
    match_type: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalUploadPasswordHistory {
    password: Option<String>,
    last_used_date: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalUploadCard {
    cardholder_name: Option<String>,
    brand: Option<String>,
    number: Option<String>,
    exp_month: Option<String>,
    exp_year: Option<String>,
    code: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalUploadIdentity {
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
struct CanonicalUploadSecureNote {
    note_type: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalUploadSshKey {
    private_key: Option<String>,
    public_key: Option<String>,
    fingerprint: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalUploadAttachment {
    file_name: String,
    size: Option<String>,
}

#[derive(Debug)]
struct PreparedUploadItem {
    name: String,
    deleted_date: Option<String>,
    payload: Value,
    item_key: SymmetricKey,
    source: DecryptedCipher,
    attachments: Vec<DecryptedAttachment>,
}

fn prepare_upload_item(
    item: &DecryptedCipher,
    folder_id: Option<String>,
    target_key: &SymmetricKey,
) -> PreparedUploadItem {
    let item_key = generate_random_symmetric_key();
    let fields = encrypt_fields(&item.fields, &item_key);
    let login = encrypt_login(item.login.as_ref(), &item_key);
    let card = encrypt_card(item.card.as_ref(), &item_key);
    let identity = encrypt_identity(item.identity.as_ref(), &item_key);
    let ssh_key = encrypt_ssh_key(item.ssh_key.as_ref(), &item_key);
    let secure_note = encrypt_secure_note(item.secure_note.as_ref());
    let notes = item.notes.as_deref().map(|value| encrypt_string_value(value, &item_key));
    let name = encrypt_string_value(&item.name, &item_key);
    let key = encrypt_string_value_raw(&item_key.to_bytes(), target_key);
    let attachments = item.attachments.clone();

    let payload = json!({
        "type": item.r#type,
        "name": name,
        "notes": notes,
        "favorite": item.favorite,
        "folderId": folder_id,
        "organizationId": Value::Null,
        "key": key,
        "fields": fields,
        "login": login,
        "card": card,
        "identity": identity,
        "secureNote": secure_note,
        "sshKey": ssh_key,
        "passwordHistory": encrypt_password_history(item.login.as_ref(), &item_key),
        "reprompt": Value::Null,
        "lastKnownRevisionDate": Value::Null,
    });

    PreparedUploadItem {
        name: item.name.clone(),
        deleted_date: item.deleted_date.clone(),
        payload,
        item_key,
        source: item.clone(),
        attachments,
    }
}

async fn load_upload_source(input_dir: &Path) -> Result<UploadSource, Error> {
    let vault_path = input_dir.join("decrypted").join("vault.json");
    let vault_bytes = fs::read(&vault_path)
        .await
        .map_err(|e| Error::new(format!("Failed to read {}", vault_path.display()), e.to_string()))?;
    let mut export: DecryptedExport =
        serde_json::from_slice(&vault_bytes).map_res("Failed to parse decrypted vault export")?;
    export.items.iter_mut().for_each(normalize_loaded_cipher);

    let attachment_index_path = input_dir.join("decrypted").join("attachments-index.json");
    let attachment_index =
        if fs::try_exists(&attachment_index_path).await.map_res("Failed to probe decrypted attachment index")? {
            let bytes = fs::read(&attachment_index_path).await.map_res("Failed to read decrypted attachment index")?;
            let entries: Vec<DecryptedAttachmentIndexEntry> =
                serde_json::from_slice(&bytes).map_res("Failed to parse decrypted attachment index")?;
            entries
                .into_iter()
                .map(|entry| ((entry.cipher_id.clone(), entry.attachment_id.clone()), entry))
                .collect::<HashMap<_, _>>()
        } else {
            HashMap::new()
        };

    Ok(UploadSource {
        export,
        attachment_index,
    })
}

fn normalize_loaded_cipher(item: &mut DecryptedCipher) {
    if item.login.as_ref().is_some_and(login_is_empty) {
        item.login = None;
    }
    if item.card.as_ref().is_some_and(card_is_empty) {
        item.card = None;
    }
    if item.identity.as_ref().is_some_and(identity_is_empty) {
        item.identity = None;
    }
    if item.secure_note.as_ref().is_some_and(secure_note_is_empty) {
        item.secure_note = None;
    }
    if item.ssh_key.as_ref().is_some_and(ssh_key_is_empty) {
        item.ssh_key = None;
    }

    match item.r#type {
        1 => {
            item.card = None;
            item.identity = None;
            item.secure_note = None;
            item.ssh_key = None;
        }
        2 => {
            item.login = None;
            item.card = None;
            item.identity = None;
            item.ssh_key = None;
        }
        3 => {
            item.login = None;
            item.identity = None;
            item.secure_note = None;
            item.ssh_key = None;
        }
        4 => {
            item.login = None;
            item.card = None;
            item.secure_note = None;
            item.ssh_key = None;
        }
        5 => {
            item.login = None;
            item.card = None;
            item.identity = None;
            item.secure_note = None;
        }
        _ => {}
    }
}

fn login_is_empty(login: &DecryptedLogin) -> bool {
    login.username.is_none()
        && login.password.is_none()
        && login.totp.is_none()
        && login.uris.is_empty()
        && login.password_history.is_empty()
}

fn card_is_empty(card: &DecryptedCard) -> bool {
    card.cardholder_name.is_none()
        && card.brand.is_none()
        && card.number.is_none()
        && card.exp_month.is_none()
        && card.exp_year.is_none()
        && card.code.is_none()
}

fn identity_is_empty(identity: &DecryptedIdentity) -> bool {
    identity.title.is_none()
        && identity.first_name.is_none()
        && identity.middle_name.is_none()
        && identity.last_name.is_none()
        && identity.address1.is_none()
        && identity.address2.is_none()
        && identity.address3.is_none()
        && identity.city.is_none()
        && identity.state.is_none()
        && identity.postal_code.is_none()
        && identity.country.is_none()
        && identity.company.is_none()
        && identity.email.is_none()
        && identity.phone.is_none()
        && identity.ssn.is_none()
        && identity.username.is_none()
        && identity.passport_number.is_none()
        && identity.license_number.is_none()
}

fn secure_note_is_empty(note: &DecryptedSecureNote) -> bool {
    note.note_type.is_none()
}

fn ssh_key_is_empty(ssh_key: &DecryptedSshKey) -> bool {
    ssh_key.private_key.is_none() && ssh_key.public_key.is_none() && ssh_key.fingerprint.is_none()
}

async fn ensure_target_folders(
    session: &Session,
    source_folders: &[DecryptedFolder],
    target_folders: &[DecryptedFolder],
) -> Result<HashMap<String, String>, Error> {
    let mut target_by_name =
        target_folders.iter().map(|folder| (folder.name.clone(), folder.id.clone())).collect::<HashMap<_, _>>();
    let mut source_to_target = HashMap::new();

    for folder in source_folders {
        let target_id = if let Some(existing) = target_by_name.get(&folder.name) {
            existing.clone()
        } else {
            let response = post_json(
                session,
                &format!("{}/api/folders", session.common.server),
                &json!({ "name": folder.name }),
                "upload",
            )
            .await?;
            let created: Value = response.json().await.map_res("Failed to parse created folder response")?;
            let id = created
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .map_res("Created folder response missing id")?;
            target_by_name.insert(folder.name.clone(), id.clone());
            id
        };
        source_to_target.insert(folder.id.clone(), target_id);
    }

    Ok(source_to_target)
}

fn folder_name_map(folders: &[DecryptedFolder]) -> HashMap<String, String> {
    folders.iter().map(|folder| (folder.id.clone(), folder.name.clone())).collect()
}

fn build_existing_signatures(
    export: &DecryptedExport,
    target_folders: &[DecryptedFolder],
) -> HashMap<String, Vec<DecryptedCipher>> {
    let target_folder_names = folder_name_map(target_folders);
    let mut signatures = HashMap::<String, Vec<DecryptedCipher>>::new();
    for item in &export.items {
        let folder_name = item.folder_id.as_ref().and_then(|folder_id| target_folder_names.get(folder_id)).cloned();
        signatures
            .entry(build_duplicate_signature(item, folder_name.as_deref()))
            .or_default()
            .push(item.clone());
    }
    signatures
}

fn insert_existing_signature(signatures: &mut HashMap<String, Vec<DecryptedCipher>>, item: DecryptedCipher) {
    signatures
        .entry(build_duplicate_signature(&item, None))
        .or_default()
        .push(item);
}

fn replace_existing_signature(
    signatures: &mut HashMap<String, Vec<DecryptedCipher>>,
    target_id: &str,
    replacement: DecryptedCipher,
) {
    for values in signatures.values_mut() {
        values.retain(|item| item.id != target_id);
    }
    insert_existing_signature(signatures, replacement);
}

fn build_duplicate_signature(item: &DecryptedCipher, folder_name_override: Option<&str>) -> String {
    let signature = CanonicalUploadCipher {
        r#type: item.r#type,
        name: item.name.clone(),
        notes: item.notes.clone(),
        favorite: item.favorite,
        folder: folder_name_override.map(str::to_owned),
        deleted: item.deleted_date.is_some(),
        fields: canonical_fields(&item.fields),
        login: item.login.as_ref().map(canonical_login),
        card: item.card.as_ref().map(canonical_card),
        identity: item.identity.as_ref().map(canonical_identity),
        secure_note: item.secure_note.as_ref().map(canonical_secure_note),
        ssh_key: item.ssh_key.as_ref().map(canonical_ssh_key),
        attachments: canonical_attachments(&item.attachments),
    };

    serde_json::to_string(&signature).unwrap_or_else(|_| item.id.clone())
}

fn resolve_source_duplicates(
    items: &[DecryptedCipher],
    folder_names: &HashMap<String, String>,
    common: &CommonArgs,
) -> Result<Vec<DecryptedCipher>, Error> {
    let mut grouped = HashMap::<String, Vec<DecryptedCipher>>::new();
    for item in items {
        let folder_name = item.folder_id.as_ref().and_then(|folder_id| folder_names.get(folder_id)).cloned();
        grouped
            .entry(build_duplicate_signature(item, folder_name.as_deref()))
            .or_default()
            .push(item.clone());
    }

    let mut resolved = Vec::with_capacity(items.len());
    for group in grouped.into_values() {
        if group.len() == 1 {
            resolved.extend(group);
        } else {
            resolved.push(resolve_source_duplicate_group(&group, common)?);
        }
    }
    Ok(resolved)
}

fn resolve_source_duplicate_group(items: &[DecryptedCipher], common: &CommonArgs) -> Result<DecryptedCipher, Error> {
    log_verbose(
        "upload",
        common,
        format!("Resolving {} duplicate source items", items.len()),
    );
    let mut terminal = setup_tui_terminal()?;
    let _guard = TuiTerminalGuard;
    let mut selected = 0usize;

    loop {
        terminal
            .draw(|frame| render_source_duplicate_picker(frame, items, selected))
            .map_res("Failed to render source duplicate picker")?;

        if !event::poll(Duration::from_millis(250)).map_res("Failed to poll duplicate picker events")? {
            continue;
        }

        let Event::Key(key) = event::read().map_res("Failed to read duplicate picker event")? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(items.len().saturating_sub(1));
            }
            KeyCode::Enter => return Ok(items[selected].clone()),
            KeyCode::Esc | KeyCode::Char('q') => {
                return Err(Error::new("Duplicate selection aborted", "source duplicate resolution cancelled"));
            }
            _ => {}
        }
    }
}

fn resolve_target_duplicate(
    source: &DecryptedCipher,
    targets: &[DecryptedCipher],
    common: &CommonArgs,
) -> Result<(UploadMode, Option<DecryptedCipher>), Error> {
    log_verbose(
        "upload",
        common,
        format!("Resolving target duplicate for {} against {} candidates", source.name, targets.len()),
    );

    let mut terminal = setup_tui_terminal()?;
    let _guard = TuiTerminalGuard;
    let mut state = DuplicatePickerState {
        source_selected: 0,
        target_selected: 0,
        focus: DuplicatePickerFocus::Source,
    };
    let sources = vec![source.clone()];

    loop {
        terminal
            .draw(|frame| render_target_duplicate_picker(frame, &sources, targets, &state))
            .map_res("Failed to render target duplicate picker")?;

        if !event::poll(Duration::from_millis(250)).map_res("Failed to poll duplicate picker events")? {
            continue;
        }

        let Event::Key(key) = event::read().map_res("Failed to read duplicate picker event")? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Tab | KeyCode::Right => {
                state.focus = match state.focus {
                    DuplicatePickerFocus::Source => DuplicatePickerFocus::Target,
                    DuplicatePickerFocus::Target => DuplicatePickerFocus::Details,
                    DuplicatePickerFocus::Details => DuplicatePickerFocus::Source,
                };
            }
            KeyCode::BackTab | KeyCode::Left => {
                state.focus = match state.focus {
                    DuplicatePickerFocus::Source => DuplicatePickerFocus::Details,
                    DuplicatePickerFocus::Target => DuplicatePickerFocus::Source,
                    DuplicatePickerFocus::Details => DuplicatePickerFocus::Target,
                };
            }
            KeyCode::Up | KeyCode::Char('k') => match state.focus {
                DuplicatePickerFocus::Source => state.source_selected = state.source_selected.saturating_sub(1),
                DuplicatePickerFocus::Target => state.target_selected = state.target_selected.saturating_sub(1),
                DuplicatePickerFocus::Details => {}
            },
            KeyCode::Down | KeyCode::Char('j') => match state.focus {
                DuplicatePickerFocus::Source => {
                    state.source_selected = (state.source_selected + 1).min(sources.len().saturating_sub(1));
                }
                DuplicatePickerFocus::Target => {
                    state.target_selected = (state.target_selected + 1).min(targets.len().saturating_sub(1));
                }
                DuplicatePickerFocus::Details => {}
            },
            KeyCode::Char('c') => return Ok((UploadMode::Create, None)),
            KeyCode::Char('r') | KeyCode::Enter => {
                return Ok((UploadMode::Replace, Some(targets[state.target_selected].clone())));
            }
            KeyCode::Char('s') => return Ok((UploadMode::Skip, None)),
            KeyCode::Esc | KeyCode::Char('q') => {
                return Err(Error::new("Duplicate selection aborted", "target duplicate resolution cancelled"));
            }
            _ => {}
        }
    }
}

async fn upload_cipher(session: &Session, item: &PreparedUploadItem) -> Result<String, Error> {
    let response =
        post_json(session, &format!("{}/api/ciphers", session.common.server), &item.payload, "upload").await?;
    let body: Value = response.json().await.map_res("Failed to parse cipher create response")?;
    let id = body.get("id").and_then(Value::as_str).map(str::to_owned).map_res("Cipher create response missing id")?;
    Ok(id)
}

async fn update_cipher(session: &Session, existing: &DecryptedCipher, item: &PreparedUploadItem) -> Result<(), Error> {
    let mut payload = item.payload.clone();
    payload["lastKnownRevisionDate"] = existing
        .revision_date
        .as_ref()
        .map(|value| Value::String(value.clone()))
        .unwrap_or(Value::Null);
    let response = put_json(
        session,
        &format!("{}/api/ciphers/{}", session.common.server, existing.id),
        &payload,
        "upload",
    )
    .await?;
    response.error_for_status().map_res("Failed to update existing cipher")?;
    Ok(())
}

async fn soft_delete_cipher(session: &Session, cipher_id: &str) -> Result<(), Error> {
    let response =
        put_json(session, &format!("{}/api/ciphers/{cipher_id}/delete", session.common.server), &json!({}), "upload")
            .await?;
    response.error_for_status().map_res("Failed to soft-delete uploaded cipher")?;
    Ok(())
}

async fn restore_cipher(session: &Session, cipher_id: &str) -> Result<(), Error> {
    let response =
        put_json(session, &format!("{}/api/ciphers/{cipher_id}/restore", session.common.server), &json!({}), "upload")
            .await?;
    response.error_for_status().map_res("Failed to restore cipher")?;
    Ok(())
}

async fn upload_cipher_attachments(
    session: &Session,
    cipher_id: &str,
    item: &PreparedUploadItem,
    attachment_index: &HashMap<(String, String), DecryptedAttachmentIndexEntry>,
    input_dir: &Path,
) -> Result<(), Error> {
    for attachment in &item.attachments {
        upload_one_attachment(
            session,
            cipher_id,
            &item.source.id,
            &item.item_key,
            attachment,
            attachment_index,
            input_dir,
        )
        .await?;
    }

    Ok(())
}

fn local_attachment_path(input_dir: &Path, entry: &DecryptedAttachmentIndexEntry) -> PathBuf {
    let path = PathBuf::from(&entry.output_path);
    if path.is_absolute() {
        path
    } else {
        input_dir.join(path)
    }
}

fn canonical_fields(fields: &[DecryptedField]) -> Vec<CanonicalUploadField> {
    fields
        .iter()
        .map(|field| CanonicalUploadField {
            name: field.name.clone(),
            value: field.value.clone(),
            field_type: field.field_type,
            linked_id: field.linked_id,
        })
        .collect()
}

fn canonical_login(login: &DecryptedLogin) -> CanonicalUploadLogin {
    CanonicalUploadLogin {
        username: login.username.clone(),
        password: login.password.clone(),
        totp: login.totp.clone(),
        uris: login
            .uris
            .iter()
            .map(|uri| CanonicalUploadUri {
                uri: uri.uri.clone(),
                match_type: uri.match_type,
            })
            .collect(),
        password_history: login
            .password_history
            .iter()
            .map(|entry| CanonicalUploadPasswordHistory {
                password: entry.password.clone(),
                last_used_date: entry.last_used_date.clone(),
            })
            .collect(),
    }
}

fn canonical_card(card: &DecryptedCard) -> CanonicalUploadCard {
    CanonicalUploadCard {
        cardholder_name: card.cardholder_name.clone(),
        brand: card.brand.clone(),
        number: card.number.clone(),
        exp_month: card.exp_month.clone(),
        exp_year: card.exp_year.clone(),
        code: card.code.clone(),
    }
}

fn canonical_identity(identity: &DecryptedIdentity) -> CanonicalUploadIdentity {
    CanonicalUploadIdentity {
        title: identity.title.clone(),
        first_name: identity.first_name.clone(),
        middle_name: identity.middle_name.clone(),
        last_name: identity.last_name.clone(),
        address1: identity.address1.clone(),
        address2: identity.address2.clone(),
        address3: identity.address3.clone(),
        city: identity.city.clone(),
        state: identity.state.clone(),
        postal_code: identity.postal_code.clone(),
        country: identity.country.clone(),
        company: identity.company.clone(),
        email: identity.email.clone(),
        phone: identity.phone.clone(),
        ssn: identity.ssn.clone(),
        username: identity.username.clone(),
        passport_number: identity.passport_number.clone(),
        license_number: identity.license_number.clone(),
    }
}

fn canonical_secure_note(note: &DecryptedSecureNote) -> CanonicalUploadSecureNote {
    CanonicalUploadSecureNote { note_type: note.note_type }
}

fn canonical_ssh_key(ssh_key: &DecryptedSshKey) -> CanonicalUploadSshKey {
    CanonicalUploadSshKey {
        private_key: ssh_key.private_key.clone(),
        public_key: ssh_key.public_key.clone(),
        fingerprint: ssh_key.fingerprint.clone(),
    }
}

fn canonical_attachments(attachments: &[DecryptedAttachment]) -> Vec<CanonicalUploadAttachment> {
    attachments
        .iter()
        .map(|attachment| CanonicalUploadAttachment {
            file_name: attachment.file_name.clone(),
            size: attachment.size.clone(),
        })
        .collect()
}

fn validation_from_cipher(item: &DecryptedCipher, folder_id: Option<String>) -> ValidationCipher {
    ValidationCipher {
        r#type: item.r#type,
        name: item.name.clone(),
        notes: item.notes.clone(),
        favorite: item.favorite,
        folder_id,
        deleted: item.deleted_date.is_some(),
        fields: canonical_fields(&item.fields),
        login: item.login.as_ref().map(canonical_login),
        card: item.card.as_ref().map(canonical_card),
        identity: item.identity.as_ref().map(canonical_identity),
        secure_note: item.secure_note.as_ref().map(canonical_secure_note),
        ssh_key: item.ssh_key.as_ref().map(canonical_ssh_key),
        attachments: canonical_attachments(&item.attachments),
    }
}

fn validation_from_prepared(item: &PreparedUploadItem) -> Result<ValidationCipher, Error> {
    let source = &item.source;
    let payload = &item.payload;
    let login = decrypt_login(payload.get("login"), payload.get("passwordHistory"), &item.item_key)?;
    let card = decrypt_card(payload.get("card"), &item.item_key)?;
    let identity = decrypt_identity(payload.get("identity"), &item.item_key)?;
    let secure_note = decrypt_secure_note(payload.get("secureNote"))?;
    let ssh_key = decrypt_ssh_key(payload.get("sshKey"), &item.item_key)?;

    Ok(ValidationCipher {
        r#type: payload.get("type").and_then(Value::as_i64).unwrap_or(source.r#type),
        name: decrypt_value_string(payload.get("name"), &item.item_key)?.unwrap_or_default(),
        notes: decrypt_value_string(payload.get("notes"), &item.item_key)?,
        favorite: payload.get("favorite").and_then(Value::as_bool),
        folder_id: payload.get("folderId").and_then(Value::as_str).map(str::to_owned),
        deleted: item.deleted_date.is_some(),
        fields: canonical_fields(&decrypt_fields(payload.get("fields"), &item.item_key)?),
        login: login.as_ref().map(canonical_login),
        card: card.as_ref().map(canonical_card),
        identity: identity.as_ref().map(canonical_identity),
        secure_note: secure_note.as_ref().map(canonical_secure_note),
        ssh_key: ssh_key.as_ref().map(canonical_ssh_key),
        attachments: canonical_attachments(&item.attachments),
    })
}

fn verify_prepared_upload_item(item: &PreparedUploadItem) -> Result<(), Error> {
    let expected = validation_from_cipher(&item.source, item.payload.get("folderId").and_then(Value::as_str).map(str::to_owned));
    let prepared = validation_from_prepared(item)?;
    let expected_json = serde_json::to_value(&expected).unwrap_or_default();
    let prepared_json = serde_json::to_value(&prepared).unwrap_or_default();
    if expected_json != prepared_json {
        return Err(Error::new(
            "Prepared upload item failed local round-trip validation",
            format!("{} expected={} actual={}", item.source.name, expected_json, prepared_json),
        ));
    }
    Ok(())
}

async fn fetch_cipher_details(session: &Session, cipher_id: &str) -> Result<Value, Error> {
    let url = format!("{}/api/ciphers/{cipher_id}", session.common.server);
    let response = session
        .client
        .get(url)
        .header(AUTHORIZATION, format!("Bearer {}", session.token.access_token))
        .send()
        .await
        .map_res("Failed to fetch cipher details")?;
    ensure_success_response(response, "Cipher fetch request failed", "upload", &session.common)
        .await?
        .json()
        .await
        .map_res("Failed to decode cipher details response")
}

async fn verify_uploaded_cipher(
    session: &Session,
    keys: &AccountKeys,
    upload_item: &PreparedUploadItem,
    cipher_id: &str,
    attachment_mode: AttachmentRestoreMode,
) -> Result<DecryptedCipher, Error> {
    let cipher_json = fetch_cipher_details(session, cipher_id).await?;
    let saved = decrypt_cipher(&cipher_json, keys, &session.common)?;
    let expected_full = validation_from_cipher(
        &upload_item.source,
        upload_item.payload.get("folderId").and_then(Value::as_str).map(str::to_owned),
    );
    let actual_full = validation_from_cipher(&saved, saved.folder_id.clone());
    let mut expected = expected_full.clone();
    let mut actual = actual_full.clone();
    if attachment_mode == AttachmentRestoreMode::SkipValidation {
        expected.attachments.clear();
        actual.attachments.clear();
    }
    if serde_json::to_value(&expected).unwrap_or_default() != serde_json::to_value(&actual).unwrap_or_default() {
        return Err(Error::new(
            "Uploaded cipher failed round-trip validation",
            format!(
                "cipher={} source={} expected={} actual={}",
                cipher_id,
                upload_item.source.name,
                serde_json::to_value(&expected).unwrap_or_default(),
                serde_json::to_value(&actual).unwrap_or_default()
            ),
        ));
    }
    Ok(saved)
}

async fn verify_uploaded_cipher_with_attachment_prompt(
    session: &Session,
    keys: &AccountKeys,
    upload_item: &PreparedUploadItem,
    cipher_id: &str,
    attachment_mode: &mut AttachmentRestoreMode,
) -> Result<DecryptedCipher, Error> {
    match verify_uploaded_cipher(session, keys, upload_item, cipher_id, *attachment_mode).await {
        Ok(saved) => Ok(saved),
        Err(err) if *attachment_mode == AttachmentRestoreMode::Enforce && attachment_validation_only(&err) => {
            if prompt_skip_attachments_on_error(&upload_item.name, &err)? {
                *attachment_mode = AttachmentRestoreMode::SkipValidation;
                verify_uploaded_cipher(session, keys, upload_item, cipher_id, *attachment_mode).await
            } else {
                Err(err)
            }
        }
        Err(err) => Err(err),
    }
}

fn attachment_validation_only(err: &Error) -> bool {
    let text = format!("{err}");
    text.contains("\"attachments\":[]") && text.contains("\"file_name\":")
}

async fn delete_cipher_attachment(session: &Session, cipher_id: &str, attachment_id: &str) -> Result<(), Error> {
    let response = post_json(
        session,
        &format!(
            "{}/api/ciphers/{cipher_id}/attachment/{attachment_id}/delete",
            session.common.server
        ),
        &json!({}),
        "upload",
    )
    .await?;
    response.error_for_status().map_res("Failed to delete target attachment")?;
    Ok(())
}

async fn reconcile_cipher_attachments(
    session: &Session,
    upload_item: &PreparedUploadItem,
    existing: &DecryptedCipher,
    attachment_index: &HashMap<(String, String), DecryptedAttachmentIndexEntry>,
    input_dir: &Path,
) -> Result<(), Error> {
    let mut existing_by_name = existing
        .attachments
        .iter()
        .map(|attachment| (attachment.file_name.clone(), attachment.clone()))
        .collect::<HashMap<_, _>>();

    for attachment in &upload_item.attachments {
        match existing_by_name.remove(&attachment.file_name) {
            Some(target_attachment) if target_attachment.size == attachment.size => {}
            Some(target_attachment) => {
                delete_cipher_attachment(session, &existing.id, &target_attachment.id).await?;
                upload_one_attachment(session, &existing.id, &upload_item.source.id, &upload_item.item_key, attachment, attachment_index, input_dir)
                    .await?;
            }
            None => {
                upload_one_attachment(session, &existing.id, &upload_item.source.id, &upload_item.item_key, attachment, attachment_index, input_dir)
                    .await?;
            }
        }
    }

    for stale in existing_by_name.into_values() {
        delete_cipher_attachment(session, &existing.id, &stale.id).await?;
    }
    Ok(())
}

async fn upload_one_attachment(
    session: &Session,
    cipher_id: &str,
    source_cipher_id: &str,
    item_key: &SymmetricKey,
    attachment: &DecryptedAttachment,
    attachment_index: &HashMap<(String, String), DecryptedAttachmentIndexEntry>,
    input_dir: &Path,
) -> Result<(), Error> {
    let key = (source_cipher_id.to_owned(), attachment.id.clone());
    let index_entry = attachment_index.get(&key).cloned().ok_or_else(|| {
        Error::new(
            "Missing decrypted attachment index entry",
            format!("{source_cipher_id} / {}", attachment.id),
        )
    })?;
    let source_path = local_attachment_path(input_dir, &index_entry);
    let bytes = fs::read(&source_path).await.map_err(|e| {
        Error::new(format!("Failed to read decrypted attachment {}", source_path.display()), e.to_string())
    })?;
    let attachment_key = generate_random_symmetric_key();
    let encrypted_name = encrypt_string_value(&attachment.file_name, item_key);
    let encrypted_key = encrypt_string_value_raw(&attachment_key.to_bytes(), item_key);
    let encrypted_bytes = encrypt_bytes_with_key(&bytes, &attachment_key)?;
    let file_size = attachment.size.clone().unwrap_or_else(|| bytes.len().to_string());

    let response = post_json(
        session,
        &format!("{}/api/ciphers/{cipher_id}/attachment/v2", session.common.server),
        &json!({
            "key": encrypted_key,
            "fileName": encrypted_name,
            "fileSize": file_size,
            "adminRequest": false,
        }),
        "upload",
    )
    .await?;
    let response_json: Value = response.json().await.map_res("Failed to parse attachment create response")?;
    let attachment_id = response_json
        .get("attachmentId")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .map_res("Attachment create response missing attachmentId")?;
    let upload_url = response_json
        .get("url")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .map_res("Attachment create response missing url")?;
    let upload_url = normalize_attachment_upload_url(&upload_url);

    let part = Part::bytes(encrypted_bytes).file_name(attachment.file_name.clone());
    let form = Form::new().part("data", part);
    let response = session
        .client
        .post(format!("{}{}", session.common.server, upload_url))
        .header(AUTHORIZATION, format!("Bearer {}", session.token.access_token))
        .multipart(form)
        .send()
        .await
        .map_res("Failed to upload encrypted attachment data")?;
    ensure_success_response(response, "Attachment upload request failed", "upload", &session.common).await?;

    log_verbose(
        "upload",
        &session.common,
        format!("Uploaded attachment {} for cipher {}", attachment_id, cipher_id),
    );
    Ok(())
}

fn normalize_attachment_upload_url(upload_url: &str) -> String {
    if upload_url.starts_with("/api/") {
        upload_url.to_owned()
    } else if upload_url.starts_with("/ciphers/") {
        format!("/api{upload_url}")
    } else {
        upload_url.to_owned()
    }
}

fn prompt_skip_attachments_on_error(item_name: &str, err: &Error) -> Result<bool, Error> {
    eprintln!("\nAttachment upload failed for item: {item_name}");
    eprintln!("{err}");
    eprintln!("Choose how to continue:");
    eprintln!("  [s] skip all remaining attachment uploads and continue restoring items");
    eprintln!("  [a] abort upload");

    loop {
        eprint!("Continue [s/a]: ");
        io::stdout().flush().map_res("Failed to flush attachment prompt")?;
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| Error::new("Failed to read attachment prompt input", e.to_string()))?;
        match input.trim().to_ascii_lowercase().as_str() {
            "s" | "skip" => return Ok(true),
            "a" | "abort" => return Ok(false),
            _ => eprintln!("Please enter s or a."),
        }
    }
}

fn encrypt_fields(fields: &[DecryptedField], key: &SymmetricKey) -> Option<Value> {
    if fields.is_empty() {
        return None;
    }

    Some(Value::Array(
        fields
            .iter()
            .map(|field| {
                json!({
                    "name": field.name.as_deref().map(|value| encrypt_string_value(value, key)),
                    "value": field.value.as_deref().map(|value| encrypt_string_value(value, key)),
                    "type": field.field_type,
                    "linkedId": field.linked_id,
                })
            })
            .collect(),
    ))
}

fn encrypt_login(login: Option<&DecryptedLogin>, key: &SymmetricKey) -> Option<Value> {
    let login = login?;
    Some(json!({
        "username": login.username.as_deref().map(|value| encrypt_string_value(value, key)),
        "password": login.password.as_deref().map(|value| encrypt_string_value(value, key)),
        "totp": login.totp.as_deref().map(|value| encrypt_string_value(value, key)),
        "uris": login.uris.iter().map(|uri| json!({
            "uri": uri.uri.as_deref().map(|value| encrypt_string_value(value, key)),
            "match": uri.match_type,
        })).collect::<Vec<_>>(),
    }))
}

fn encrypt_password_history(login: Option<&DecryptedLogin>, key: &SymmetricKey) -> Option<Value> {
    let login = login?;
    if login.password_history.is_empty() {
        return None;
    }

    Some(Value::Array(
        login
            .password_history
            .iter()
            .map(|entry| {
                json!({
                    "password": entry.password.as_deref().map(|value| encrypt_string_value(value, key)),
                    "lastUsedDate": entry.last_used_date.clone(),
                })
            })
            .collect(),
    ))
}

fn encrypt_card(card: Option<&DecryptedCard>, key: &SymmetricKey) -> Option<Value> {
    let card = card?;
    Some(json!({
        "cardholderName": card.cardholder_name.as_deref().map(|value| encrypt_string_value(value, key)),
        "brand": card.brand.as_deref().map(|value| encrypt_string_value(value, key)),
        "number": card.number.as_deref().map(|value| encrypt_string_value(value, key)),
        "expMonth": card.exp_month.as_deref().map(|value| encrypt_string_value(value, key)),
        "expYear": card.exp_year.as_deref().map(|value| encrypt_string_value(value, key)),
        "code": card.code.as_deref().map(|value| encrypt_string_value(value, key)),
    }))
}

fn encrypt_identity(identity: Option<&DecryptedIdentity>, key: &SymmetricKey) -> Option<Value> {
    let identity = identity?;
    Some(json!({
        "title": identity.title.as_deref().map(|value| encrypt_string_value(value, key)),
        "firstName": identity.first_name.as_deref().map(|value| encrypt_string_value(value, key)),
        "middleName": identity.middle_name.as_deref().map(|value| encrypt_string_value(value, key)),
        "lastName": identity.last_name.as_deref().map(|value| encrypt_string_value(value, key)),
        "address1": identity.address1.as_deref().map(|value| encrypt_string_value(value, key)),
        "address2": identity.address2.as_deref().map(|value| encrypt_string_value(value, key)),
        "address3": identity.address3.as_deref().map(|value| encrypt_string_value(value, key)),
        "city": identity.city.as_deref().map(|value| encrypt_string_value(value, key)),
        "state": identity.state.as_deref().map(|value| encrypt_string_value(value, key)),
        "postalCode": identity.postal_code.as_deref().map(|value| encrypt_string_value(value, key)),
        "country": identity.country.as_deref().map(|value| encrypt_string_value(value, key)),
        "company": identity.company.as_deref().map(|value| encrypt_string_value(value, key)),
        "email": identity.email.as_deref().map(|value| encrypt_string_value(value, key)),
        "phone": identity.phone.as_deref().map(|value| encrypt_string_value(value, key)),
        "ssn": identity.ssn.as_deref().map(|value| encrypt_string_value(value, key)),
        "username": identity.username.as_deref().map(|value| encrypt_string_value(value, key)),
        "passportNumber": identity.passport_number.as_deref().map(|value| encrypt_string_value(value, key)),
        "licenseNumber": identity.license_number.as_deref().map(|value| encrypt_string_value(value, key)),
    }))
}

fn encrypt_secure_note(note: Option<&DecryptedSecureNote>) -> Option<Value> {
    note.map(|note| {
        json!({
            "type": note.note_type,
        })
    })
}

fn encrypt_ssh_key(ssh_key: Option<&DecryptedSshKey>, key: &SymmetricKey) -> Option<Value> {
    let ssh_key = ssh_key?;
    Some(json!({
        "privateKey": ssh_key.private_key.as_deref().map(|value| encrypt_string_value(value, key)),
        "publicKey": ssh_key.public_key.as_deref().map(|value| encrypt_string_value(value, key)),
        "fingerprint": ssh_key.fingerprint.as_deref().map(|value| encrypt_string_value(value, key)),
    }))
}

fn encrypt_string_value(value: &str, key: &SymmetricKey) -> String {
    encrypt_cipher_string_to_string(value.as_bytes(), key)
}

fn encrypt_string_value_raw(value: &[u8], key: &SymmetricKey) -> String {
    encrypt_cipher_string_to_string(value, key)
}

fn encrypt_cipher_string_to_string(plaintext: &[u8], key: &SymmetricKey) -> String {
    let iv = crypto::get_random_bytes::<16>();
    let ciphertext = encrypt(Cipher::aes_256_cbc(), &key.enc, Some(&iv), plaintext).expect("AES-CBC encryption failed");
    if let Some(mac_key) = &key.mac {
        let mut input = Vec::with_capacity(iv.len() + ciphertext.len());
        input.extend_from_slice(&iv);
        input.extend_from_slice(&ciphertext);
        let mac = hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, mac_key), &input);
        format!("2.{}|{}|{}", BASE64.encode(&iv), BASE64.encode(&ciphertext), BASE64.encode(mac.as_ref()))
    } else {
        format!("0.{}|{}", BASE64.encode(&iv), BASE64.encode(&ciphertext))
    }
}

fn encrypt_bytes_with_key(plaintext: &[u8], key: &SymmetricKey) -> Result<Vec<u8>, Error> {
    let iv = crypto::get_random_bytes::<16>();
    let ciphertext =
        encrypt(Cipher::aes_256_cbc(), &key.enc, Some(&iv), plaintext).map_res("AES-CBC encryption failed")?;
    let mut output = Vec::with_capacity(1 + iv.len() + ciphertext.len() + 32);
    if let Some(mac_key) = &key.mac {
        let mut input = Vec::with_capacity(iv.len() + ciphertext.len());
        input.extend_from_slice(&iv);
        input.extend_from_slice(&ciphertext);
        let mac = hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, mac_key), &input);
        output.push(2);
        output.extend_from_slice(&iv);
        output.extend_from_slice(mac.as_ref());
        output.extend_from_slice(&ciphertext);
    } else {
        output.push(0);
        output.extend_from_slice(&iv);
        output.extend_from_slice(&ciphertext);
    }
    Ok(output)
}

fn generate_random_symmetric_key() -> SymmetricKey {
    SymmetricKey::from_bytes(crypto::get_random_bytes::<64>().to_vec()).expect("generated symmetric key length")
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

    push_optional_line(&mut lines, "folderId", item.folder_id.clone());
    push_optional_line(&mut lines, "deletedDate", item.deleted_date.clone());
    push_optional_line(&mut lines, "creationDate", item.creation_date.clone());
    push_optional_line(&mut lines, "revisionDate", item.revision_date.clone());

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
        for history in &login.password_history {
            let password = history.password.clone().unwrap_or_default();
            let last_used = history.last_used_date.clone().unwrap_or_default();
            lines.push(Line::from(format!("history: {password} [{last_used}]")));
        }
    }

    if let Some(card) = &item.card {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled("card", Style::default().add_modifier(Modifier::BOLD))));
        push_optional_line(&mut lines, "cardholder", card.cardholder_name.clone());
        push_optional_line(&mut lines, "brand", card.brand.clone());
        push_optional_line(&mut lines, "number", card.number.clone());
        push_optional_line(&mut lines, "expMonth", card.exp_month.clone());
        push_optional_line(&mut lines, "expYear", card.exp_year.clone());
        push_optional_line(&mut lines, "code", card.code.clone());
    }

    if let Some(identity) = &item.identity {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled("identity", Style::default().add_modifier(Modifier::BOLD))));
        push_optional_line(&mut lines, "title", identity.title.clone());
        push_optional_line(&mut lines, "firstName", identity.first_name.clone());
        push_optional_line(&mut lines, "middleName", identity.middle_name.clone());
        push_optional_line(&mut lines, "lastName", identity.last_name.clone());
        push_optional_line(&mut lines, "address1", identity.address1.clone());
        push_optional_line(&mut lines, "address2", identity.address2.clone());
        push_optional_line(&mut lines, "address3", identity.address3.clone());
        push_optional_line(&mut lines, "city", identity.city.clone());
        push_optional_line(&mut lines, "state", identity.state.clone());
        push_optional_line(&mut lines, "postalCode", identity.postal_code.clone());
        push_optional_line(&mut lines, "country", identity.country.clone());
        push_optional_line(&mut lines, "company", identity.company.clone());
        push_optional_line(&mut lines, "email", identity.email.clone());
        push_optional_line(&mut lines, "phone", identity.phone.clone());
        push_optional_line(&mut lines, "ssn", identity.ssn.clone());
        push_optional_line(&mut lines, "username", identity.username.clone());
        push_optional_line(&mut lines, "passport", identity.passport_number.clone());
        push_optional_line(&mut lines, "license", identity.license_number.clone());
    }

    if let Some(ssh_key) = &item.ssh_key {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled("sshKey", Style::default().add_modifier(Modifier::BOLD))));
        push_optional_line(&mut lines, "privateKey", ssh_key.private_key.clone());
        push_optional_line(&mut lines, "publicKey", ssh_key.public_key.clone());
        push_optional_line(&mut lines, "fingerprint", ssh_key.fingerprint.clone());
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

fn render_source_duplicate_picker(frame: &mut Frame<'_>, items: &[DecryptedCipher], selected: usize) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(frame.size());
    let help = Paragraph::new("Duplicate source items: use ↑/↓ to inspect, Enter to keep one, q to cancel")
        .block(Block::default().borders(Borders::ALL).title("Upload Duplicate Resolution"));
    frame.render_widget(help, root[0]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(34), Constraint::Percentage(66)])
        .split(root[1]);

    let items_list = items
        .iter()
        .enumerate()
        .map(|(idx, item)| ListItem::new(format!("{} {}", idx + 1, item.name)))
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected(Some(selected));
    let list = List::new(items_list)
        .block(Block::default().borders(Borders::ALL).title("Source Candidates"))
        .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");
    frame.render_stateful_widget(list, body[0], &mut state);

    let lines = build_item_detail_lines(&items[selected]);
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Details"))
            .wrap(Wrap { trim: false }),
        body[1],
    );
}

fn render_target_duplicate_picker(
    frame: &mut Frame<'_>,
    sources: &[DecryptedCipher],
    targets: &[DecryptedCipher],
    state: &DuplicatePickerState,
) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(frame.size());
    let help =
        Paragraph::new("Source vs target duplicate: Tab switches panes, c=create duplicate, r/Enter=replace target, s=skip, q=cancel")
            .block(Block::default().borders(Borders::ALL).title("Upload Duplicate Resolution"));
    frame.render_widget(help, root[0]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(25), Constraint::Percentage(50)])
        .split(root[1]);

    let source_items = sources
        .iter()
        .map(|item| ListItem::new(item.name.clone()))
        .collect::<Vec<_>>();
    let target_items = targets
        .iter()
        .map(|item| ListItem::new(format!("{} ({})", item.name, item.id)))
        .collect::<Vec<_>>();

    let source_title = if state.focus == DuplicatePickerFocus::Source {
        "Source [focused]"
    } else {
        "Source"
    };
    let target_title = if state.focus == DuplicatePickerFocus::Target {
        "Target [focused]"
    } else {
        "Target"
    };
    let details_title = if state.focus == DuplicatePickerFocus::Details {
        "Details [focused]"
    } else {
        "Details"
    };

    let mut source_state = ListState::default().with_selected(Some(state.source_selected));
    frame.render_stateful_widget(
        List::new(source_items)
            .block(Block::default().borders(Borders::ALL).title(source_title))
            .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD))
            .highlight_symbol(">> "),
        body[0],
        &mut source_state,
    );

    let mut target_state = ListState::default().with_selected(Some(state.target_selected));
    frame.render_stateful_widget(
        List::new(target_items)
            .block(Block::default().borders(Borders::ALL).title(target_title))
            .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD))
            .highlight_symbol(">> "),
        body[1],
        &mut target_state,
    );

    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled("Incoming Source", Style::default().add_modifier(Modifier::BOLD))));
    lines.extend(build_item_detail_lines(&sources[state.source_selected]));
    lines.push(Line::default());
    lines.push(Line::from(Span::styled("Existing Target", Style::default().add_modifier(Modifier::BOLD))));
    lines.extend(build_item_detail_lines(&targets[state.target_selected]));
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(details_title))
            .wrap(Wrap { trim: false }),
        body[2],
    );
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

fn decrypt_export_lossy(sync: &Value, keys: &AccountKeys, common: &CommonArgs) -> DecryptedExport {
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
        match decrypt_cipher(cipher, keys, common) {
            Ok(item) => items.push(item),
            Err(err) => {
                let cipher_id = cipher.get("id").and_then(Value::as_str).unwrap_or("<unknown>");
                let cipher_type = cipher.get("type").and_then(Value::as_i64).unwrap_or_default();
                log_info(
                    "upload",
                    common,
                    format!("Skipping undecryptable target cipher id={cipher_id} type={} reason={err}", cipher_type),
                );
            }
        }
    }

    DecryptedExport {
        encrypted: false,
        folders,
        items,
    }
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
    if login.is_null() {
        return Ok(None);
    }

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
    if card.is_null() {
        return Ok(None);
    }

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
    if identity.is_null() {
        return Ok(None);
    }

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
    if note.is_null() {
        return Ok(None);
    }
    Ok(Some(DecryptedSecureNote {
        note_type: note.get("type").and_then(Value::as_i64),
    }))
}

fn decrypt_ssh_key(value: Option<&Value>, key: &SymmetricKey) -> Result<Option<DecryptedSshKey>, Error> {
    let Some(ssh_key) = value else {
        return Ok(None);
    };
    if ssh_key.is_null() {
        return Ok(None);
    }

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

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.enc.clone();
        if let Some(mac) = &self.mac {
            bytes.extend_from_slice(mac);
        }
        bytes
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

async fn post_json(session: &Session, url: &str, body: &Value, tag: &str) -> Result<reqwest::Response, Error> {
    let response = session
        .client
        .post(url)
        .header(AUTHORIZATION, format!("Bearer {}", session.token.access_token))
        .json(body)
        .send()
        .await
        .map_res("Failed to send POST request")?;
    ensure_success_response(response, "POST request failed", tag, &session.common).await
}

async fn put_json(session: &Session, url: &str, body: &Value, tag: &str) -> Result<reqwest::Response, Error> {
    let response = session
        .client
        .put(url)
        .header(AUTHORIZATION, format!("Bearer {}", session.token.access_token))
        .json(body)
        .send()
        .await
        .map_res("Failed to send PUT request")?;
    ensure_success_response(response, "PUT request failed", tag, &session.common).await
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
        build_existing_signatures, collect_attachment_downloads, decrypt_aes_cbc, decrypt_bytes_with_key,
        decrypt_cipher_string_to_bytes, generate_totp_code, hkdf_expand_from_prk_32, normalize_base_url,
        normalize_client_id, normalize_uri_host, normalize_attachment_upload_url, parse_cipher_string,
        parse_totp_value, path_to_forward_slashes, prepare_upload_item, sanitize_file_name,
        verify_prepared_upload_item, DecryptedCipher, DecryptedExport, DecryptedField, DecryptedFolder,
        DecryptedLogin, DecryptedPasswordHistory, DecryptedUri, SymmetricKey, TotpAlgorithm,
        seconds_remaining_in_period,
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

    fn sample_cipher(id: &str, folder_id: &str) -> DecryptedCipher {
        DecryptedCipher {
            id: id.to_owned(),
            r#type: 1,
            name: "Example".to_owned(),
            notes: Some("notes".to_owned()),
            favorite: Some(true),
            folder_id: Some(folder_id.to_owned()),
            organization_id: None,
            collection_ids: Vec::new(),
            deleted_date: None,
            creation_date: None,
            revision_date: None,
            fields: vec![DecryptedField {
                name: Some("field".to_owned()),
                value: Some("value".to_owned()),
                field_type: Some(1),
                linked_id: None,
            }],
            attachments: Vec::new(),
            login: Some(DecryptedLogin {
                username: Some("user@example.com".to_owned()),
                password: Some("super-secret".to_owned()),
                totp: None,
                uris: vec![DecryptedUri {
                    uri: Some("https://example.com".to_owned()),
                    match_type: Some(0),
                }],
                password_history: vec![DecryptedPasswordHistory {
                    password: Some("old-secret".to_owned()),
                    last_used_date: Some("2024-01-01T00:00:00.000000Z".to_owned()),
                }],
            }),
            card: None,
            identity: None,
            secure_note: None,
            ssh_key: None,
            search_blob: String::new(),
            search_domains: Vec::new(),
        }
    }

    #[test]
    fn build_existing_signatures_groups_duplicates() {
        let folder = DecryptedFolder {
            id: "folder-1".to_owned(),
            name: "Logins".to_owned(),
        };
        let export = DecryptedExport {
            encrypted: false,
            folders: vec![folder.clone()],
            items: vec![sample_cipher("a", &folder.id), sample_cipher("b", &folder.id)],
        };

        let signatures = build_existing_signatures(&export, &[folder]);
        assert_eq!(signatures.len(), 1);
        assert_eq!(signatures.values().next().unwrap().len(), 2);
    }

    #[test]
    fn prepare_upload_item_round_trips_locally() {
        let source = sample_cipher("cipher-1", "folder-1");
        let key = SymmetricKey::from_bytes((0..64).map(|i| i as u8).collect()).unwrap();
        let upload = prepare_upload_item(&source, Some("target-folder".to_owned()), &key);
        verify_prepared_upload_item(&upload).unwrap();
    }

    #[test]
    fn normalize_attachment_upload_url_prefixes_api_mount() {
        assert_eq!(
            normalize_attachment_upload_url("/ciphers/abc/attachment/def"),
            "/api/ciphers/abc/attachment/def"
        );
        assert_eq!(
            normalize_attachment_upload_url("/api/ciphers/abc/attachment/def"),
            "/api/ciphers/abc/attachment/def"
        );
    }
}
