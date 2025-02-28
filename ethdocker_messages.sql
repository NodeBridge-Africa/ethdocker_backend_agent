-- Create the messages table for ETHDocker conversations
create table if not exists ethdocker_messages (
    id bigserial primary key,
    session_id varchar not null,
    message jsonb not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add indexes for better query performance
    constraint ethdocker_messages_session_id_idx unique (session_id, id)
);

-- Create index on session_id for faster lookups
create index if not exists ethdocker_messages_session_lookup 
on ethdocker_messages(session_id);

-- Create index on created_at for time-based queries
create index if not exists ethdocker_messages_timestamp_lookup 
on ethdocker_messages(created_at);

-- Enable RLS
alter table ethdocker_messages enable row level security;

-- Create policy that allows anyone to read
create policy "Allow public read access"
  on ethdocker_messages
  for select
  to public
  using (true);

-- Create policy that allows authenticated users to insert
create policy "Allow authenticated insert"
  on ethdocker_messages
  for insert
  to authenticated
  with check (true); 